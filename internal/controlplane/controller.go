// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package controlplane

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/leaderelection"
	rl "k8s.io/client-go/tools/leaderelection/resourcelock"

	"github.com/NVIDIA/k8s-test-infra/internal/mokkacontroller"
	versioned "github.com/NVIDIA/k8s-test-infra/pkg/generated/clientset/versioned"
)

// Controller owns the elected controller lifecycle and readiness state.
type Controller struct {
	config     Config
	kubeClient kubernetes.Interface
	reconciler *mokkacontroller.Controller
	readiness  *electionReadiness
}

// NewController builds Kubernetes clients and the informer-driven reconciler.
func NewController(config Config) (*Controller, error) {
	if err := ValidateControllerConfig(config); err != nil {
		return nil, err
	}
	restConfig, err := controllerRESTConfig(config)
	if err != nil {
		return nil, err
	}
	kubeClient, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("create Kubernetes client: %w", err)
	}
	mokkaClient, err := versioned.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("create Mokka client: %w", err)
	}
	reconciler, err := mokkacontroller.New(kubeClient, mokkaClient, mokkacontroller.Options{
		Workers: config.Workers, StatusDebounce: config.StatusDebounce,
		StatusProgressInterval: config.StatusProgressInterval,
		LiveNodeGetTimeout:     config.LiveNodeGetTimeout,
	})
	if err != nil {
		return nil, fmt.Errorf("create Mokka controller: %w", err)
	}
	return &Controller{
		config: config, kubeClient: kubeClient, reconciler: reconciler,
		readiness: newElectionReadiness(),
	}, nil
}

// ValidateControllerConfig rejects settings that cannot make progress safely.
//
//nolint:cyclop // Each branch reports a distinct unsafe controller setting.
func ValidateControllerConfig(config Config) error {
	if config.LeaderElectionNamespace == "" || config.LeaderElectionName == "" {
		return errors.New("leader-election namespace and name must not be empty")
	}
	if config.Workers < 1 || config.StatusDebounce < 0 || config.StatusProgressInterval < 0 ||
		config.LiveNodeGetTimeout <= 0 {
		return errors.New("workers and live Node GET timeout must be positive and status intervals non-negative")
	}
	if config.StatusProgressInterval > 0 && config.StatusProgressInterval < config.StatusDebounce {
		return errors.New("status progress interval must not be shorter than status debounce")
	}
	if config.LeaseDuration <= 0 || config.RenewDeadline <= 0 || config.RetryPeriod <= 0 ||
		config.LeaseDuration <= config.RenewDeadline ||
		config.RenewDeadline <= time.Duration(leaderelection.JitterFactor*float64(config.RetryPeriod)) {
		return errors.New("leader-election durations must satisfy lease > renew > retry*jitter")
	}
	if config.KubeAPIQPS <= 0 || config.KubeAPIBurst < 1 {
		return errors.New("kubernetes API QPS and burst must be positive")
	}
	return nil
}

// Ready reports whether this replica can participate in controller service.
// Standbys become ready after observing the Lease; a leader additionally waits
// for every informer cache to synchronize.
func (c *Controller) Ready() bool {
	return c != nil && c.reconciler != nil && c.readiness != nil && c.readiness.ready(c.reconciler.Ready())
}

// Run participates in Lease election and runs workers only while leading.
func (c *Controller) Run(ctx context.Context) error {
	identity, err := leaderIdentity()
	if err != nil {
		return err
	}
	lock := &rl.LeaseLock{
		LeaseMeta: metav1.ObjectMeta{
			Name: c.config.LeaderElectionName, Namespace: c.config.LeaderElectionNamespace,
		},
		Client:     c.kubeClient.CoordinationV1(),
		LockConfig: rl.ResourceLockConfig{Identity: identity},
	}
	return runLeaderElection(ctx, c.config, lock, c.reconciler.Run, c.readiness)
}

func controllerRESTConfig(config Config) (*rest.Config, error) {
	var (
		result *rest.Config
		err    error
	)
	if config.Kubeconfig != "" {
		result, err = clientcmd.BuildConfigFromFlags("", config.Kubeconfig)
	} else {
		result, err = rest.InClusterConfig()
	}
	if err != nil {
		return nil, fmt.Errorf("load Kubernetes configuration: %w", err)
	}
	result = rest.CopyConfig(result)
	result.QPS = float32(config.KubeAPIQPS)
	result.Burst = config.KubeAPIBurst
	result.UserAgent = "mokka-control-plane"
	return result, nil
}

func leaderIdentity() (string, error) {
	hostname, err := os.Hostname()
	if err != nil {
		return "", fmt.Errorf("get hostname for leader identity: %w", err)
	}
	return hostname + "_" + uuid.NewString(), nil
}

func newLeaderElectionConfig(
	config Config,
	lock rl.Interface,
	run func(context.Context),
	readiness *electionReadiness,
) leaderelection.LeaderElectionConfig {
	onStartedLeading := run
	onStoppedLeading := func() {}
	var onNewLeader func(string)
	if readiness != nil {
		onStartedLeading = func(ctx context.Context) {
			readiness.startLeading()
			run(ctx)
		}
		onStoppedLeading = readiness.stop
		onNewLeader = func(identity string) {
			readiness.observeLeader(identity == lock.Identity())
		}
	}
	return leaderelection.LeaderElectionConfig{
		Lock: lock, LeaseDuration: config.LeaseDuration, RenewDeadline: config.RenewDeadline,
		RetryPeriod: config.RetryPeriod, ReleaseOnCancel: true, Name: config.LeaderElectionName,
		Callbacks: leaderelection.LeaderCallbacks{
			OnStartedLeading: onStartedLeading,
			OnStoppedLeading: onStoppedLeading,
			OnNewLeader:      onNewLeader,
		},
	}
}

func runLeaderElection(
	ctx context.Context,
	config Config,
	lock rl.Interface,
	run func(context.Context) error,
	readiness *electionReadiness,
) error {
	if readiness != nil {
		readiness.start()
		defer readiness.stop()
	}
	electionCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	work := newLeaderWork()
	draining := &drainingLock{Interface: lock, workDone: work.done, stopWork: cancel}
	electionConfig := newLeaderElectionConfig(
		config, draining, work.onStartedLeading(run, cancel), readiness,
	)
	elector, err := leaderelection.NewLeaderElector(electionConfig)
	if err != nil {
		return fmt.Errorf("configure leader election: %w", err)
	}
	elector.Run(electionCtx)
	cancel()
	if !work.finishElection() {
		return nil
	}
	<-work.done
	return work.result()
}

type electionReadinessState uint8

const (
	electionStopped electionReadinessState = iota
	electionWaiting
	electionStandby
	electionLeader
)

type electionReadiness struct {
	state atomic.Uint32
}

func newElectionReadiness() *electionReadiness {
	return &electionReadiness{}
}

func (r *electionReadiness) start() {
	r.state.Store(uint32(electionWaiting))
}

func (r *electionReadiness) observeLeader(self bool) {
	if self {
		r.startLeading()
		return
	}
	// OnNewLeader callbacks run asynchronously. Once this replica is elected,
	// a delayed observation of the previous leader must not mark it standby.
	r.state.CompareAndSwap(uint32(electionWaiting), uint32(electionStandby))
}

func (r *electionReadiness) startLeading() {
	for {
		state := electionReadinessState(r.state.Load())
		switch state {
		case electionWaiting, electionStandby:
			if r.state.CompareAndSwap(uint32(state), uint32(electionLeader)) {
				return
			}
		case electionStopped, electionLeader:
			return
		}
	}
}

func (r *electionReadiness) stop() {
	r.state.Store(uint32(electionStopped))
}

func (r *electionReadiness) ready(leaderReady bool) bool {
	switch electionReadinessState(r.state.Load()) {
	case electionStandby:
		return true
	case electionLeader:
		return leaderReady
	default:
		return false
	}
}

type leaderWork struct {
	mu               sync.Mutex
	done             chan struct{}
	electionFinished bool
	started          bool
	err              error
}

func newLeaderWork() *leaderWork {
	return &leaderWork{done: make(chan struct{})}
}

func (w *leaderWork) onStartedLeading(
	run func(context.Context) error,
	stopElection context.CancelFunc,
) func(context.Context) {
	return func(ctx context.Context) {
		w.start(ctx, run, stopElection)
	}
}

func (w *leaderWork) start(ctx context.Context, run func(context.Context) error, stopElection context.CancelFunc) {
	w.mu.Lock()
	if w.electionFinished {
		w.mu.Unlock()
		return
	}
	w.started = true
	w.mu.Unlock()

	err := run(ctx)
	w.mu.Lock()
	w.err = err
	w.mu.Unlock()
	close(w.done)
	stopElection()
}

func (w *leaderWork) finishElection() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.electionFinished = true
	return w.started
}

func (w *leaderWork) result() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.err
}

type drainingLock struct {
	rl.Interface
	workDone <-chan struct{}
	stopWork context.CancelFunc
}

func (l *drainingLock) Update(ctx context.Context, record rl.LeaderElectionRecord) error {
	if record.HolderIdentity == "" {
		if l.stopWork != nil {
			l.stopWork()
		}
		select {
		case <-l.workDone:
		case <-ctx.Done():
			return context.Cause(ctx)
		}
	}
	return l.Interface.Update(ctx, record)
}
