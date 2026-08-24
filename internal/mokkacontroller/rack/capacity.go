// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

package rack

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"sync/atomic"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	mokkav1alpha1 "github.com/NVIDIA/k8s-test-infra/internal/controlplane/api/v1alpha1"
)

const (
	// MaxInventoryNodes is the largest admitted controller topology.
	MaxInventoryNodes int64 = 100_000
	// ReasonCapacityExceeded identifies declarations outside the supported topology envelope.
	ReasonCapacityExceeded = "CapacityExceeded"
)

type capacityRevision uint64

type admissionInventory struct {
	instance inventoryInstance
	created  metav1.Time
	capacity DeclaredCapacity
}

type capacityAdmissionSnapshot struct {
	revision capacityRevision
	admitted []admissionInventory
	err      error
}

// CapacityAdmission coalesces deterministic admission across concurrent
// workers. Its published slice contains one entry per positive-capacity
// admitted Inventory, bounded by the rack limit, and retains no rejected set.
// Candidates are considered by creation timestamp, name, and UID; each valid
// materializable Inventory is admitted whole when it fits the remaining limit.
type CapacityAdmission struct {
	cache Cache

	revision     atomic.Uint64
	computations atomic.Uint64
	mu           sync.Mutex
	snapshot     *capacityAdmissionSnapshot
}

// NewCapacityAdmission constructs aggregate admission over informer state.
func NewCapacityAdmission(cache Cache) *CapacityAdmission {
	return &CapacityAdmission{cache: cache}
}

// Invalidate prevents workers using the previous Inventory/Profile snapshot
// from continuing rack-proportional work.
func (a *CapacityAdmission) Invalidate() {
	a.revision.Add(1)
}

func (a *CapacityAdmission) currentRevision() capacityRevision {
	return capacityRevision(a.revision.Load())
}

func (a *CapacityAdmission) current(revision capacityRevision) bool {
	return revision == a.currentRevision()
}

func (a *CapacityAdmission) admits(
	revision capacityRevision,
	inventory *mokkav1alpha1.SGPUInventory,
	capacity DeclaredCapacity,
) (bool, error) {
	if !a.current(revision) {
		return false, errAllocationInputChanged
	}
	if capacity.Racks == 0 {
		return true, nil
	}
	snapshot := a.snapshotFor(revision)
	if snapshot.err != nil {
		return false, snapshot.err
	}
	if !a.current(revision) {
		return false, errAllocationInputChanged
	}
	candidate := admissionInventory{
		instance: inventoryInstance{name: inventory.Name, uid: inventory.UID},
		created:  inventory.CreationTimestamp,
		capacity: capacity,
	}
	index, found := slices.BinarySearchFunc(snapshot.admitted, candidate, compareAdmissionInventories)
	if !found {
		return false, nil
	}
	admitted := snapshot.admitted[index]
	if admitted.capacity != candidate.capacity {
		return false, errAllocationInputChanged
	}
	return true, nil
}

func (a *CapacityAdmission) snapshotFor(revision capacityRevision) *capacityAdmissionSnapshot {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.snapshot != nil && a.snapshot.revision == revision {
		return a.snapshot
	}
	if !a.current(revision) {
		return &capacityAdmissionSnapshot{revision: revision, err: errAllocationInputChanged}
	}
	snapshot := a.computeSnapshot(revision)
	if !a.current(revision) {
		return &capacityAdmissionSnapshot{revision: revision, err: errAllocationInputChanged}
	}
	if snapshot.err == nil {
		a.computations.Add(1)
	}
	a.snapshot = snapshot
	return snapshot
}

func (a *CapacityAdmission) computeSnapshot(revision capacityRevision) *capacityAdmissionSnapshot {
	inventories, err := a.cache.Inventories()
	if err != nil {
		return &capacityAdmissionSnapshot{
			revision: revision,
			err:      fmt.Errorf("list inventories for capacity admission: %w", err),
		}
	}
	slices.SortFunc(inventories, compareInventoryAdmissionOrder)
	admitted := make([]admissionInventory, 0, min(len(inventories), int(MaxInventoryNodes)))
	total := DeclaredCapacity{}
	for _, inventory := range inventories {
		capacity, materializes, capacityErr := materializedInventoryCapacity(a.cache, inventory)
		if capacityErr != nil {
			return &capacityAdmissionSnapshot{revision: revision, err: capacityErr}
		}
		if !materializes {
			continue
		}
		candidate := admissionInventory{
			instance: inventoryInstance{name: inventory.Name, uid: inventory.UID},
			created:  inventory.CreationTimestamp,
			capacity: capacity,
		}
		next, addErr := AddCapacity(total, capacity)
		if addErr != nil || ValidateSupportedCapacity(next) != nil {
			continue
		}
		admitted = append(admitted, candidate)
		total = next
	}
	return &capacityAdmissionSnapshot{revision: revision, admitted: admitted}
}

func materializedInventoryCapacity(
	cache Cache,
	inventory *mokkav1alpha1.SGPUInventory,
) (DeclaredCapacity, bool, error) {
	resolved, err := materializedInventoryGroups(cache, inventory)
	if err != nil {
		return DeclaredCapacity{}, false, err
	}
	total, err := capacityForResolvedGroups(resolved)
	return total, len(resolved) > 0, err
}

func materializedInventoryGroups(
	cache Cache,
	inventory *mokkav1alpha1.SGPUInventory,
) ([]resolvedGroup, error) {
	if inventory == nil || inventory.DeletionTimestamp != nil || validateInventory(inventory) != nil ||
		validateInventoryRackCapacity(inventory) != nil {
		return nil, nil
	}
	resolved, _, err := (&Reconciler{cache: cache}).resolveGroups(inventory)
	if err != nil {
		return nil, err
	}
	if validateResolvedCapacity(resolved) != nil {
		return nil, nil
	}
	resolved, _ = validateGroupMaterialization(inventory, resolved)
	return resolved, nil
}

func admitInventoryCapacities(candidates []admissionInventory) []admissionInventory {
	admitted := make([]admissionInventory, 0, min(len(candidates), int(MaxInventoryNodes)))
	total := DeclaredCapacity{}
	for _, candidate := range candidates {
		next, err := AddCapacity(total, candidate.capacity)
		if err != nil || ValidateSupportedCapacity(next) != nil {
			continue
		}
		admitted = append(admitted, candidate)
		total = next
	}
	return admitted
}

func compareInventoryAdmissionOrder(a, b *mokkav1alpha1.SGPUInventory) int {
	return compareAdmissionInventories(
		admissionInventory{instance: inventoryInstance{name: a.Name, uid: a.UID}, created: a.CreationTimestamp},
		admissionInventory{instance: inventoryInstance{name: b.Name, uid: b.UID}, created: b.CreationTimestamp},
	)
}

func compareAdmissionInventories(a, b admissionInventory) int {
	if order := a.created.Time.Compare(b.created.Time); order != 0 {
		return order
	}
	if order := cmp.Compare(a.instance.name, b.instance.name); order != 0 {
		return order
	}
	return cmp.Compare(string(a.instance.uid), string(b.instance.uid))
}

func aggregateCapacityAdmissionError(inventory *mokkav1alpha1.SGPUInventory) string {
	return fmt.Sprintf(
		"inventory %q is outside the aggregate limit of %d Nodes or racks; Inventories are admitted whole by oldest-first fit",
		inventory.Name,
		MaxInventoryNodes,
	)
}

// DeclaredCapacity holds checked capacity values before conversion to API status types.
type DeclaredCapacity struct {
	Racks int64
	Nodes int64
	GPUs  int64
}

func validateInventoryRackCapacity(inventory *mokkav1alpha1.SGPUInventory) error {
	// Reject aggregate-invalid declarations before profile resolution or any
	// work proportional to the declared rack count.
	total := DeclaredCapacity{}
	for _, group := range inventory.Spec.RackGroups {
		var err error
		total, err = AddCapacity(total, DeclaredCapacity{Racks: int64(group.Count)})
		if err != nil {
			return err
		}
	}
	return ValidateSupportedCapacity(total)
}

// CapacityForGroup computes one group's declared capacity with checked intermediates.
func CapacityForGroup(group mokkav1alpha1.RackGroup, profile *mokkav1alpha1.SGPURackProfile) (DeclaredCapacity, error) {
	if profile == nil {
		return DeclaredCapacity{}, fmt.Errorf("rack group %q profile must not be nil", group.ID)
	}
	racks := int64(group.Count)
	nodes, ok := checkedMultiply(racks, int64(profile.Spec.Rack.NodesPerRack))
	if !ok {
		return DeclaredCapacity{}, fmt.Errorf("rack group %q Node capacity overflows int64", group.ID)
	}
	gpus, ok := checkedMultiply(nodes, int64(profile.Spec.Node.GPUs.Count))
	if !ok {
		return DeclaredCapacity{}, fmt.Errorf("rack group %q GPU capacity overflows int64", group.ID)
	}
	return DeclaredCapacity{Racks: racks, Nodes: nodes, GPUs: gpus}, nil
}

// AddCapacity combines checked group or inventory capacity values.
func AddCapacity(a, b DeclaredCapacity) (DeclaredCapacity, error) {
	racks, ok := checkedAdd(a.Racks, b.Racks)
	if !ok {
		return DeclaredCapacity{}, errors.New("aggregate rack capacity overflows int64")
	}
	nodes, ok := checkedAdd(a.Nodes, b.Nodes)
	if !ok {
		return DeclaredCapacity{}, errors.New("aggregate Node capacity overflows int64")
	}
	gpus, ok := checkedAdd(a.GPUs, b.GPUs)
	if !ok {
		return DeclaredCapacity{}, errors.New("aggregate GPU capacity overflows int64")
	}
	return DeclaredCapacity{Racks: racks, Nodes: nodes, GPUs: gpus}, nil
}

// ValidateSupportedCapacity enforces the controller scale contract and status bounds.
func ValidateSupportedCapacity(capacity DeclaredCapacity) error {
	if capacity.Racks < 0 || capacity.Nodes < 0 || capacity.GPUs < 0 {
		return errors.New("declared capacity must not be negative")
	}
	if capacity.Nodes > MaxInventoryNodes {
		return fmt.Errorf(
			"desired Nodes %d exceed supported maximum %d",
			capacity.Nodes,
			MaxInventoryNodes,
		)
	}
	if capacity.Racks > MaxInventoryNodes {
		return fmt.Errorf("desired racks %d exceed supported maximum %d", capacity.Racks, MaxInventoryNodes)
	}
	if capacity.Racks > math.MaxInt32 || capacity.Nodes > math.MaxInt32 || capacity.GPUs > math.MaxInt32 {
		return errors.New("declared capacity exceeds int32 status bounds")
	}
	return nil
}

// StatusCapacity converts capacity only after the supported bounds are established.
func StatusCapacity(capacity DeclaredCapacity) (mokkav1alpha1.InventoryCapacity, error) {
	if err := ValidateSupportedCapacity(capacity); err != nil {
		return mokkav1alpha1.InventoryCapacity{}, err
	}
	return mokkav1alpha1.InventoryCapacity{
		Racks: int32(capacity.Racks),
		Nodes: int32(capacity.Nodes),
		GPUs:  int32(capacity.GPUs),
	}, nil
}

func checkedMultiply(a, b int64) (int64, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > math.MaxInt64/a {
		return 0, false
	}
	return a * b, true
}

func checkedAdd(a, b int64) (int64, bool) {
	if a < 0 || b < 0 || b > math.MaxInt64-a {
		return 0, false
	}
	return a + b, true
}
