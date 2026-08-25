// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package rdmaplugin

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"

	"github.com/NVIDIA/k8s-test-infra/internal/agent"
)

const testNode = "worker-0"

// fakeNodes is a minimal stand-in for the node API. client-go's fake clientset
// is not vendored, and applying the two JSON Patch ops this simulator emits is
// cheap enough that the fake can reflect them in its node — so tests assert on
// the resulting capacity, not only on the wire bytes.
type fakeNodes struct {
	mu       sync.Mutex
	capacity map[corev1.ResourceName]resource.Quantity
	missing  bool // Get returns NotFound
	patches  [][]patchOp
	getErr   error
}

func newFakeNodes() *fakeNodes {
	return &fakeNodes{capacity: map[corev1.ResourceName]resource.Quantity{}}
}

func (f *fakeNodes) Get(_ context.Context, name string, _ metav1.GetOptions) (*corev1.Node, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.getErr != nil {
		return nil, f.getErr
	}
	if f.missing {
		return nil, apierrors.NewNotFound(schema.GroupResource{Resource: "nodes"}, name)
	}
	capacity := corev1.ResourceList{}
	for k, v := range f.capacity {
		capacity[k] = v
	}
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status:     corev1.NodeStatus{Capacity: capacity},
	}, nil
}

func (f *fakeNodes) Patch(_ context.Context, name string, pt types.PatchType, data []byte,
	_ metav1.PatchOptions, subresources ...string,
) (*corev1.Node, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if pt != types.JSONPatchType {
		return nil, apierrors.NewBadRequest("unexpected patch type " + string(pt))
	}
	if len(subresources) != 1 || subresources[0] != "status" {
		return nil, apierrors.NewBadRequest("capacity lives on the status subresource")
	}

	var ops []patchOp
	if err := json.Unmarshal(data, &ops); err != nil {
		return nil, apierrors.NewBadRequest(err.Error())
	}
	f.patches = append(f.patches, ops)

	for _, op := range ops {
		if err := f.applyOp(op); err != nil {
			return nil, err
		}
	}
	return &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}}, nil
}

// applyOp mutates the fake's capacity so tests assert on the resulting node
// rather than only on the wire bytes. Caller holds the lock.
func (f *fakeNodes) applyOp(op patchOp) error {
	key := corev1.ResourceName(unescapeCapacityPath(op.Path))
	switch op.Op {
	case "add":
		q, err := resource.ParseQuantity(op.Value)
		if err != nil {
			return apierrors.NewBadRequest(err.Error())
		}
		f.capacity[key] = q
		return nil
	case "remove":
		if _, ok := f.capacity[key]; !ok {
			// Matches the API server: removing an absent pointer is invalid.
			return apierrors.NewBadRequest("remove of missing path " + op.Path)
		}
		delete(f.capacity, key)
		return nil
	default:
		return apierrors.NewBadRequest("unexpected op " + op.Op)
	}
}

func (f *fakeNodes) quantity(t *testing.T, name string) (string, bool) {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	q, ok := f.capacity[corev1.ResourceName(name)]
	if !ok {
		return "", false
	}
	return q.String(), true
}

func (f *fakeNodes) patchCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.patches)
}

func unescapeCapacityPath(path string) string {
	key := strings.TrimPrefix(path, "/status/capacity/")
	key = strings.ReplaceAll(key, "~1", "/")
	return strings.ReplaceAll(key, "~0", "~")
}

func newSim(nodes NodeClient) *Simulator {
	s := New(nodes, slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.interval = 5 * time.Millisecond
	return s
}

func stateWith(res agent.RDMAResource) *agent.State {
	return &agent.State{
		Node:    agent.NodeMeta{NodeName: testNode},
		Network: agent.NetworkState{RDMAResource: res},
	}
}

func TestApply_AdvertisesResource(t *testing.T) {
	nodes := newFakeNodes()
	s := newSim(nodes)
	ctx := context.Background()

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{Name: "rdma/ib", Count: 64})))
	require.True(t, s.Ready())
	require.NoError(t, s.Apply(ctx, nil, nil))

	got, ok := nodes.quantity(t, "rdma/ib")
	require.True(t, ok, "rdma/ib missing from capacity")
	require.Equal(t, "64", got)

	// The slash must reach the API server as the RFC 6901 escape, or the patch
	// addresses a nested object instead of one capacity key.
	require.Equal(t, "/status/capacity/rdma~1ib", nodes.patches[0][0].Path)
}

// A node that already reports the right value must cost no write: Run re-asserts
// on every tick, so patching unconditionally would churn the node object.
func TestApply_SkipsWriteWhenAlreadyCorrect(t *testing.T) {
	nodes := newFakeNodes()
	s := newSim(nodes)
	ctx := context.Background()

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{Name: "rdma/ib", Count: 64})))
	require.NoError(t, s.Apply(ctx, nil, nil))
	require.Equal(t, 1, nodes.patchCount())

	require.NoError(t, s.Apply(ctx, nil, nil))
	require.Equal(t, 1, nodes.patchCount(), "second Apply patched an already-correct node")
}

func TestApply_NoResourceDeclared(t *testing.T) {
	nodes := newFakeNodes()
	s := newSim(nodes)
	ctx := context.Background()

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{})))
	require.True(t, s.Ready(), "a profile without the block is not an error")
	require.NoError(t, s.Apply(ctx, nil, nil))
	require.Zero(t, nodes.patchCount())
}

// Converge rather than only materialize: a profile that renames the resource
// must not leave the previous key stranded on the node.
func TestApply_WithdrawsRenamedResource(t *testing.T) {
	nodes := newFakeNodes()
	s := newSim(nodes)
	ctx := context.Background()

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{Name: "rdma/ib", Count: 64})))
	require.NoError(t, s.Apply(ctx, nil, nil))

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{Name: "rdma/roce", Count: 8})))
	require.NoError(t, s.Apply(ctx, nil, nil))

	_, ok := nodes.quantity(t, "rdma/ib")
	require.False(t, ok, "old resource still advertised")
	got, ok := nodes.quantity(t, "rdma/roce")
	require.True(t, ok)
	require.Equal(t, "8", got)
}

func TestApply_WithdrawsDroppedResource(t *testing.T) {
	nodes := newFakeNodes()
	s := newSim(nodes)
	ctx := context.Background()

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{Name: "rdma/ib", Count: 64})))
	require.NoError(t, s.Apply(ctx, nil, nil))

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{})))
	require.NoError(t, s.Apply(ctx, nil, nil))

	_, ok := nodes.quantity(t, "rdma/ib")
	require.False(t, ok)
}

func TestApply_UpdatesChangedCount(t *testing.T) {
	nodes := newFakeNodes()
	s := newSim(nodes)
	ctx := context.Background()

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{Name: "rdma/ib", Count: 64})))
	require.NoError(t, s.Apply(ctx, nil, nil))

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{Name: "rdma/ib", Count: 32})))
	require.NoError(t, s.Apply(ctx, nil, nil))

	got, _ := nodes.quantity(t, "rdma/ib")
	require.Equal(t, "32", got)
}

func TestStage_RejectsInvalidResource(t *testing.T) {
	tests := map[string]struct {
		state *agent.State
		want  string
	}{
		"unqualified name": {
			state: stateWith(agent.RDMAResource{Name: "ib", Count: 64}),
			want:  "must be qualified",
		},
		"zero count": {
			state: stateWith(agent.RDMAResource{Name: "rdma/ib"}),
			want:  "count must be positive",
		},
		"no node name": {
			state: &agent.State{Network: agent.NetworkState{
				RDMAResource: agent.RDMAResource{Name: "rdma/ib", Count: 64},
			}},
			want: "NODE_NAME is empty",
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			s := newSim(newFakeNodes())
			err := s.Stage(context.Background(), nil, tc.state)
			require.ErrorContains(t, err, tc.want)
			require.False(t, s.Ready())
		})
	}
}

func TestRevoke(t *testing.T) {
	nodes := newFakeNodes()
	s := newSim(nodes)
	ctx := context.Background()

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{Name: "rdma/ib", Count: 64})))
	require.NoError(t, s.Apply(ctx, nil, nil))
	require.NoError(t, s.Revoke(ctx, nil))

	_, ok := nodes.quantity(t, "rdma/ib")
	require.False(t, ok, "kubelet never removes a capacity key it does not own")

	// Idempotent: preStop can run after the resource is already gone.
	require.NoError(t, s.Revoke(ctx, nil))
}

func TestRevoke_NothingAdvertised(t *testing.T) {
	nodes := newFakeNodes()
	s := newSim(nodes)

	require.NoError(t, s.Revoke(context.Background(), nil))
	require.Zero(t, nodes.patchCount())
}

// A deleted node is the desired end state, not a teardown failure.
func TestRevoke_NodeGone(t *testing.T) {
	nodes := newFakeNodes()
	s := newSim(nodes)
	ctx := context.Background()

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{Name: "rdma/ib", Count: 64})))
	require.NoError(t, s.Apply(ctx, nil, nil))

	nodes.mu.Lock()
	nodes.missing = true
	nodes.mu.Unlock()

	require.NoError(t, s.Revoke(ctx, nil))
}

// The reconcile loop is level-triggered on profile content, which does not
// change when another controller overwrites the node status — so the daemon has
// to restore the resource on its own.
func TestRun_ReassertsAfterExternalRemoval(t *testing.T) {
	nodes := newFakeNodes()
	s := newSim(nodes)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	require.NoError(t, s.Stage(ctx, nil, stateWith(agent.RDMAResource{Name: "rdma/ib", Count: 64})))
	require.NoError(t, s.Apply(ctx, nil, nil))

	nodes.mu.Lock()
	delete(nodes.capacity, "rdma/ib")
	nodes.mu.Unlock()

	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	require.Eventually(t, func() bool {
		_, ok := nodes.quantity(t, "rdma/ib")
		return ok
	}, 2*time.Second, 5*time.Millisecond, "daemon never re-asserted the resource")

	cancel()
	require.NoError(t, <-done)
}

func TestRun_StopsOnContextCancel(t *testing.T) {
	s := newSim(newFakeNodes())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	require.NoError(t, s.Run(ctx))
}

func TestDiscard_TouchesNothing(t *testing.T) {
	nodes := newFakeNodes()
	s := newSim(nodes)

	require.NoError(t, s.Discard(context.Background(), nil))
	require.Zero(t, nodes.patchCount())
}
