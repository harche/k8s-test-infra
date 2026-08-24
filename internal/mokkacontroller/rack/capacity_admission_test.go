// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License").

package rack

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clienttesting "k8s.io/client-go/testing"

	mokkav1alpha1 "github.com/NVIDIA/k8s-test-infra/internal/controlplane/api/v1alpha1"
	"github.com/NVIDIA/k8s-test-infra/pkg/mokka/allocate"
)

func TestCapacityAdmissionIsDeterministicAndRecoversCapacity(t *testing.T) {
	profile := testProfile("profile", "profile-uid", 1, 1, 1)
	first := admissionTestInventory("first", "first-uid", profile.Name, 60_000, 1)
	blocked := admissionTestInventory("blocked", "blocked-uid", profile.Name, 50_000, 2)
	later := admissionTestInventory("later", "later-uid", profile.Name, 40_000, 3)
	source := &mutableAllocationSource{
		inventories: []*mokkav1alpha1.SGPUInventory{later, blocked, first},
		profiles:    map[string]*mokkav1alpha1.SGPURackProfile{profile.Name: profile},
	}

	admission := NewCapacityAdmission(source)
	revision := admission.currentRevision()
	require.True(t, capacityAdmissionDecision(t, admission, revision, first))
	require.False(t, capacityAdmissionDecision(t, admission, revision, blocked))
	require.True(t, capacityAdmissionDecision(t, admission, revision, later),
		"first-fit admission uses capacity that an older oversized declaration cannot consume")
	require.EqualValues(t, 1, admission.computations.Load())
	metadataOnly := first.DeepCopy()
	metadataOnly.ResourceVersion = "2"
	require.True(t, capacityAdmissionDecision(t, admission, revision, metadataOnly),
		"metadata-only informer updates must keep using the capacity snapshot")

	restarted := NewCapacityAdmission(source)
	require.True(t, capacityAdmissionDecision(t, restarted, restarted.currentRevision(), first))
	require.False(t, capacityAdmissionDecision(t, restarted, restarted.currentRevision(), blocked))
	require.True(t, capacityAdmissionDecision(t, restarted, restarted.currentRevision(), later))

	source.mu.Lock()
	source.inventories = []*mokkav1alpha1.SGPUInventory{blocked, later}
	source.mu.Unlock()
	admission.Invalidate()
	revision = admission.currentRevision()
	require.True(t, capacityAdmissionDecision(t, admission, revision, blocked))
	require.True(t, capacityAdmissionDecision(t, admission, revision, later))

	resized := blocked.DeepCopy()
	resized.ResourceVersion = "2"
	resized.Spec.RackGroups[0].Count = 100_000
	source.mu.Lock()
	source.inventories = []*mokkav1alpha1.SGPUInventory{resized, later}
	source.mu.Unlock()
	admission.Invalidate()
	revision = admission.currentRevision()
	require.True(t, capacityAdmissionDecision(t, admission, revision, resized))
	require.False(t, capacityAdmissionDecision(t, admission, revision, later))
}

func TestCapacityAdmissionIgnoresInvalidAndWhollyUnresolvedInventories(t *testing.T) {
	profile := testProfile("profile", "profile-uid", 1, 1, 1)
	invalidProfile := testProfile("invalid", "invalid-profile-uid", 1, 0, 1)
	invalidSpec := admissionTestInventory("invalid-spec", "invalid-spec-uid", profile.Name, 100_000, 1)
	invalidSpec.Spec.RackGroups[0].Count = 0
	missingProfile := admissionTestInventory("missing", "missing-uid", "missing", 100_000, 2)
	badProfile := admissionTestInventory("bad-profile", "bad-profile-uid", invalidProfile.Name, 100_000, 3)
	valid := admissionTestInventory("valid", "valid-uid", profile.Name, 100_000, 4)
	source := &mutableAllocationSource{
		inventories: []*mokkav1alpha1.SGPUInventory{invalidSpec, missingProfile, badProfile, valid},
		profiles: map[string]*mokkav1alpha1.SGPURackProfile{
			profile.Name: profile, invalidProfile.Name: invalidProfile,
		},
	}
	admission := NewCapacityAdmission(source)
	revision := admission.currentRevision()

	require.True(t, capacityAdmissionDecision(t, admission, revision, valid))
	snapshot := admission.snapshotFor(revision)
	require.NoError(t, snapshot.err)
	require.Equal(t, []inventoryInstance{{name: valid.Name, uid: valid.UID}}, admittedInstances(snapshot.admitted))
}

func TestCapacityAdmissionChargesOnlyMaterializableGroups(t *testing.T) {
	profile := testProfile("profile", "profile-uid", 1, 1, 1)
	partial := admissionTestInventory("partial", "partial-uid", profile.Name, 60_000, 1)
	partial.Spec.RackGroups = append(partial.Spec.RackGroups, mokkav1alpha1.RackGroup{
		ID: "unresolved", Count: 40_000,
		ProfileRef: mokkav1alpha1.ProfileReference{Name: "missing"},
	})
	follower := admissionTestInventory("follower", "follower-uid", profile.Name, 40_000, 2)
	source := &mutableAllocationSource{
		inventories: []*mokkav1alpha1.SGPUInventory{follower, partial},
		profiles:    map[string]*mokkav1alpha1.SGPURackProfile{profile.Name: profile},
	}
	admission := NewCapacityAdmission(source)
	revision := admission.currentRevision()

	partialCapacity, materializes, err := materializedInventoryCapacity(source, partial)
	require.NoError(t, err)
	require.True(t, materializes)
	require.Equal(t, DeclaredCapacity{Racks: 60_000, Nodes: 60_000, GPUs: 60_000}, partialCapacity)
	require.True(t, capacityAdmissionDecision(t, admission, revision, partial))
	require.True(t, capacityAdmissionDecision(t, admission, revision, follower))
}

func TestCapacityAdmissionCoalescesConcurrentWorkers(t *testing.T) {
	const workerCount = 64
	profile := testProfile("profile", "profile-uid", 1, 1, 1)
	inventory := admissionTestInventory("inventory", "inventory-uid", profile.Name, 100_000, 1)
	source := &mutableAllocationSource{
		inventories: []*mokkav1alpha1.SGPUInventory{inventory},
		profiles:    map[string]*mokkav1alpha1.SGPURackProfile{profile.Name: profile},
	}
	admission := NewCapacityAdmission(source)
	revision := admission.currentRevision()
	capacity, materializes, err := materializedInventoryCapacity(source, inventory)
	require.NoError(t, err)
	require.True(t, materializes)

	start := make(chan struct{})
	errors := make([]error, workerCount)
	decisions := make([]bool, workerCount)
	var workers sync.WaitGroup
	workers.Add(workerCount)
	for index := range workerCount {
		go func() {
			defer workers.Done()
			<-start
			decisions[index], errors[index] = admission.admits(revision, inventory, capacity)
		}()
	}
	close(start)
	workers.Wait()

	for index := range workerCount {
		require.NoError(t, errors[index])
		require.True(t, decisions[index])
	}
	require.EqualValues(t, 1, admission.computations.Load())
}

func TestCapacityAdmission100KBoundUsesLinearBoundedState(t *testing.T) {
	const declarationCount = MaxInventoryNodes + 1
	candidates := make([]admissionInventory, declarationCount)
	for index := range candidates {
		candidates[index] = admissionInventory{
			instance: inventoryInstance{name: fmt.Sprintf("inventory-%06d", index)},
			capacity: DeclaredCapacity{Racks: 1, Nodes: 1, GPUs: 1},
		}
	}

	admitted := admitInventoryCapacities(candidates)

	require.Len(t, admitted, int(MaxInventoryNodes))
	require.Equal(t, "inventory-099999", admitted[len(admitted)-1].instance.name)
}

func TestReconcileRejectsAggregateCrossInventoryCapacityBeforeAllocationOrWrites(t *testing.T) {
	ctx := context.Background()
	profile := testProfile("profile", "profile-uid", 1, 1, 1)
	first := admissionTestInventory("first", "first-uid", profile.Name, 60_000, 1)
	blocked := admissionTestInventory("blocked", "blocked-uid", profile.Name, 60_000, 2)
	blocked.Finalizers = []string{InventoryFinalizer}
	h := newHarness(t, []runtime.Object{profile, first, blocked}, nil)
	allocation := NewAllocationCache(h.cache)
	allocationCalls := 0
	allocation.allocate = func(allocate.Input) (allocate.Plan, error) {
		allocationCalls++
		return allocate.Plan{}, nil
	}
	reconciler := NewReconcilerWithAllocationCache(
		h.cache,
		h.mokka.MokkaV1alpha1().SGPUInventories(),
		h.mokka.MokkaV1alpha1().SGPURacks(),
		CleanupGateFunc(func(CleanupNeeded) bool { return false }),
		allocation,
	)
	h.mokka.Fake.ClearActions()

	result, err := reconciler.Reconcile(ctx, blocked.Name)

	require.NoError(t, err)
	require.False(t, result.Accepted)
	require.Equal(t, ReasonCapacityExceeded, result.ValidationReason)
	require.Contains(t, result.ValidationError, "admitted whole by oldest-first fit")
	require.Zero(t, allocationCalls)
	require.Zero(t, result.Work)
	require.Empty(t, h.mokka.Actions())
}

func TestAllocationInputExcludesAggregateRejectedInventories(t *testing.T) {
	profile := testProfile("profile", "profile-uid", 1, 1, 1)
	first := admissionTestInventory("first", "first-uid", profile.Name, 60_000, 1)
	blocked := admissionTestInventory("blocked", "blocked-uid", profile.Name, 60_000, 2)
	source := &mutableAllocationSource{
		inventories: []*mokkav1alpha1.SGPUInventory{blocked, first},
		profiles:    map[string]*mokkav1alpha1.SGPURackProfile{profile.Name: profile},
	}

	input, err := allocationInput(source)

	require.NoError(t, err)
	require.Len(t, input.Groups, 1)
	require.Equal(t, first.Name, input.Groups[0].Key.InventoryName)
}

func TestReconcileStopsStaleMaterializationAfterAdmissionChanges(t *testing.T) {
	ctx := context.Background()
	profile := testProfile("profile", "profile-uid", 1, 1, 1)
	inventory := admissionTestInventory("inventory", "inventory-uid", profile.Name, 2, 1)
	inventory.Finalizers = []string{InventoryFinalizer}
	h := newHarness(t, []runtime.Object{profile, inventory}, nil)
	allocation := NewAllocationCache(h.cache)
	reconciler := NewReconcilerWithAllocationCache(
		h.cache,
		h.mokka.MokkaV1alpha1().SGPUInventories(),
		h.mokka.MokkaV1alpha1().SGPURacks(),
		CleanupGateFunc(func(CleanupNeeded) bool { return false }),
		allocation,
	)
	created := 0
	h.mokka.Fake.PrependReactor("create", "sgpuracks", func(clienttesting.Action) (bool, runtime.Object, error) {
		created++
		if created == 1 {
			allocation.InvalidateCapacity()
		}
		return false, nil, nil
	})

	result, err := reconciler.Reconcile(ctx, inventory.Name)

	require.ErrorIs(t, err, errAllocationInputChanged)
	require.Equal(t, 1, created)
	require.EqualValues(t, 1, result.Work.RacksReconciled)
}

func capacityAdmissionDecision(
	t *testing.T,
	admission *CapacityAdmission,
	revision capacityRevision,
	inventory *mokkav1alpha1.SGPUInventory,
) bool {
	t.Helper()
	capacity, materializes, err := materializedInventoryCapacity(admission.cache, inventory)
	require.NoError(t, err)
	require.True(t, materializes)
	admitted, err := admission.admits(revision, inventory, capacity)
	require.NoError(t, err)
	return admitted
}

func admissionTestInventory(
	name string,
	uid types.UID,
	profileName string,
	count int32,
	created int64,
) *mokkav1alpha1.SGPUInventory {
	inventory := testInventory(name, uid, profileName, count)
	inventory.CreationTimestamp = metav1.NewTime(time.Unix(created, 0))
	inventory.ResourceVersion = "1"
	return inventory
}

func admittedInstances(admitted []admissionInventory) []inventoryInstance {
	instances := make([]inventoryInstance, len(admitted))
	for index := range admitted {
		instances[index] = admitted[index].instance
	}
	return instances
}
