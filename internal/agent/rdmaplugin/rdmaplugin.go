// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Package rdmaplugin simulates the RDMA shared device plugin that the Network
// Operator runs on real InfiniBand nodes. That plugin's only externally visible
// effect is a node extended resource — rdma/ib = rdmaHcaMax — so this simulator
// stages no host artifacts and publishes only that resource.
//
// The real plugin cannot run against the mock: it is a Go binary, so the
// LD_PRELOAD sysfs redirect never sees its syscalls, and it discovers HCAs
// through netdevs the mock does not render. Advertising the resource directly
// is the bridge, the same role pcibus.Apply's NFD feature file plays for PCI
// presence.
package rdmaplugin

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"github.com/NVIDIA/k8s-test-infra/internal/agent"
	"github.com/NVIDIA/k8s-test-infra/internal/agent/host"
)

const name = "rdmaplugin"

// reassertInterval bounds how long the node can go without the resource after
// something outside the agent drops it — a status overwrite by another
// controller, or an API error during Apply. The reconcile loop cannot cover
// this: it is level-triggered on profile content, which does not change.
const reassertInterval = 30 * time.Second

var (
	_ agent.Simulator = (*Simulator)(nil)
	_ agent.Applier   = (*Simulator)(nil)
	_ agent.Daemon    = (*Simulator)(nil)
)

// NodeClient is the slice of the Kubernetes node API this simulator needs.
// Satisfied by clientset.CoreV1().Nodes().
type NodeClient interface {
	Get(ctx context.Context, name string, opts metav1.GetOptions) (*corev1.Node, error)
	Patch(ctx context.Context, name string, pt types.PatchType, data []byte,
		opts metav1.PatchOptions, subresources ...string) (*corev1.Node, error)
}

// Simulator implements agent.Simulator, agent.Applier and agent.Daemon.
type Simulator struct {
	nodes    NodeClient
	log      *slog.Logger
	interval time.Duration

	ready atomic.Bool

	mu       sync.Mutex
	nodeName string
	desired  agent.RDMAResource
	// advertised is the resource name currently published, so a profile that
	// renames or drops the resource has the previous key withdrawn rather than
	// left stranded on the node.
	advertised string
}

// New returns a Simulator that publishes to nodes.
func New(nodes NodeClient, log *slog.Logger) *Simulator {
	return &Simulator{nodes: nodes, log: log, interval: reassertInterval}
}

// Name returns the simulator's stable identifier.
func (s *Simulator) Name() string { return name }

// Ready reports whether the last Stage call completed without error.
func (s *Simulator) Ready() bool { return s.ready.Load() }

// Stage resolves and validates the resource the profile asks for. There is
// nothing to materialize on the host, so the staged artifact is the validated
// desired state that Apply and Run publish.
func (s *Simulator) Stage(_ context.Context, _ *host.Host, state *agent.State) error {
	s.ready.Store(false)

	res := state.Network.RDMAResource
	if res.Name != "" {
		// Kubelet rejects an unqualified extended resource name, and a patch
		// naming one fails on the node with a message far from its cause.
		if !strings.Contains(res.Name, "/") {
			return fmt.Errorf("resource name %q must be qualified, e.g. rdma/ib", res.Name)
		}
		if res.Count <= 0 {
			return fmt.Errorf("resource %s: count must be positive, got %d", res.Name, res.Count)
		}
		if state.Node.NodeName == "" {
			return fmt.Errorf("resource %s: NODE_NAME is empty, cannot patch a node", res.Name)
		}
	}

	s.mu.Lock()
	s.nodeName, s.desired = state.Node.NodeName, res
	s.mu.Unlock()

	s.ready.Store(true)
	return nil
}

// Apply publishes the resource into the node's status capacity. Kubelet mirrors
// it into allocatable on its next status sync.
func (s *Simulator) Apply(ctx context.Context, _ *host.Host, _ *agent.State) error {
	nodeName, desired, advertised := s.snapshot()

	if advertised != "" && advertised != desired.Name {
		if err := s.withdraw(ctx, nodeName, advertised); err != nil {
			return err
		}
	}
	if desired.Name == "" {
		return nil
	}
	return s.publish(ctx, nodeName, desired)
}

// Revoke withdraws the resource so the node stops offering IB capacity it no
// longer has. Kubelet never removes a capacity key it does not own, so without
// this the resource outlives the agent.
func (s *Simulator) Revoke(ctx context.Context, _ *host.Host) error {
	nodeName, _, advertised := s.snapshot()
	if advertised == "" {
		return nil
	}
	return s.withdraw(ctx, nodeName, advertised)
}

// Discard is a no-op: this simulator writes nothing to the host.
func (s *Simulator) Discard(_ context.Context, _ *host.Host) error { return nil }

// Run re-asserts the resource on an interval and blocks until ctx is done.
func (s *Simulator) Run(ctx context.Context) error {
	t := time.NewTicker(s.interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-t.C:
			nodeName, desired, _ := s.snapshot()
			if desired.Name == "" {
				continue
			}
			if err := s.publish(ctx, nodeName, desired); err != nil {
				s.log.Warn("re-asserting rdma resource failed",
					"simulator", name, "resource", desired.Name, "err", err)
			}
		}
	}
}

// publish makes the node report the resource, reading first so a node that
// already carries the right value costs one GET rather than a write.
func (s *Simulator) publish(ctx context.Context, nodeName string, res agent.RDMAResource) error {
	node, err := s.nodes.Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get node %s: %w", nodeName, err)
	}

	want := strconv.Itoa(res.Count)
	if have, ok := node.Status.Capacity[corev1.ResourceName(res.Name)]; ok && have.String() == want {
		s.setAdvertised(res.Name)
		return nil
	}

	if err := s.patch(ctx, nodeName, patchOp{Op: "add", Path: capacityPath(res.Name), Value: want}); err != nil {
		return err
	}
	s.setAdvertised(res.Name)
	s.log.Info("advertised rdma resource", "simulator", name, "resource", res.Name, "count", res.Count)
	return nil
}

// withdraw removes the resource. A "remove" op on an absent path is rejected by
// the API server, so the key is read first; a node that already lacks it is the
// state we want.
func (s *Simulator) withdraw(ctx context.Context, nodeName, resource string) error {
	node, err := s.nodes.Get(ctx, nodeName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		s.setAdvertised("") // the node itself is gone; nothing to withdraw from
		return nil
	}
	if err != nil {
		return fmt.Errorf("get node %s: %w", nodeName, err)
	}

	if _, ok := node.Status.Capacity[corev1.ResourceName(resource)]; ok {
		if err := s.patch(ctx, nodeName, patchOp{Op: "remove", Path: capacityPath(resource)}); err != nil {
			return err
		}
		s.log.Info("withdrew rdma resource", "simulator", name, "resource", resource)
	}
	s.setAdvertised("")
	return nil
}

func (s *Simulator) patch(ctx context.Context, nodeName string, op patchOp) error {
	body, err := json.Marshal([]patchOp{op})
	if err != nil {
		return fmt.Errorf("marshal patch: %w", err)
	}
	if _, err := s.nodes.Patch(ctx, nodeName, types.JSONPatchType, body, metav1.PatchOptions{}, "status"); err != nil {
		return fmt.Errorf("patch node %s status: %w", nodeName, err)
	}
	return nil
}

func (s *Simulator) snapshot() (nodeName string, desired agent.RDMAResource, advertised string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.nodeName, s.desired, s.advertised
}

func (s *Simulator) setAdvertised(resource string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.advertised = resource
}

// patchOp is one RFC 6902 JSON Patch operation.
type patchOp struct {
	Op    string `json:"op"`
	Path  string `json:"path"`
	Value string `json:"value,omitempty"`
}

// capacityPath returns the JSON Pointer for one capacity key. Per RFC 6901 "~"
// is escaped before "/", so rdma/ib addresses a single key rather than a nested
// object that does not exist.
func capacityPath(resource string) string {
	escaped := strings.ReplaceAll(resource, "~", "~0")
	escaped = strings.ReplaceAll(escaped, "/", "~1")
	return "/status/capacity/" + escaped
}
