// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

package rack

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	mokkav1alpha1 "github.com/NVIDIA/k8s-test-infra/internal/controlplane/api/v1alpha1"
	"github.com/NVIDIA/k8s-test-infra/pkg/mokka/allocate"
	"github.com/NVIDIA/k8s-test-infra/pkg/mokka/materialize"
	"github.com/stretchr/testify/require"
)

func TestProjectionTargetAllowedRequiresOwnedDesiredAllocatedBinding(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*mokkav1alpha1.SGPURack, *corev1.Node)
		want   bool
	}{
		{name: "owned desired eligible binding", want: true},
		{
			name: "foreign rack",
			mutate: func(rack *mokkav1alpha1.SGPURack, _ *corev1.Node) {
				rack.OwnerReferences = nil
			},
		},
		{
			name: "undesired rack template",
			mutate: func(rack *mokkav1alpha1.SGPURack, _ *corev1.Node) {
				rack.Spec.Identity.FabricUUID = "caller-controlled"
			},
		},
		{
			name: "ineligible Node",
			mutate: func(_ *mokkav1alpha1.SGPURack, node *corev1.Node) {
				delete(node.Labels, allocate.EligibleNodeLabel)
			},
		},
		{
			name: "selector-mismatched Node",
			mutate: func(_ *mokkav1alpha1.SGPURack, node *corev1.Node) {
				node.Labels["pool"] = "other"
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source, inventory, keys := allocationScaleSource(1, 1)
			key := keys[0]
			profile := source.profiles[inventory.Spec.RackGroups[0].ProfileRef.Name]
			rendered, err := materialize.RenderRack(materialize.RackInput{
				InventoryName: inventory.Name,
				InventoryUID:  inventory.UID,
				Group:         inventory.Spec.RackGroups[0],
				RackIndex:     0,
				Profile:       profile,
			})
			require.NoError(t, err)
			rack := newRack(inventory, rendered.Name, rendered.Spec)
			rack.UID = "rack-uid"
			node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{
				Name: source.nodes[0].Name, UID: source.nodes[0].UID,
				Labels: map[string]string{
					allocate.EligibleNodeLabel: "true",
					"pool":                     key.RackGroup,
				},
			}}
			rack.Spec.Nodes[0].NodeRef = &mokkav1alpha1.SGPUNodeReference{Name: node.Name, UID: node.UID}
			if tt.mutate != nil {
				tt.mutate(rack, node)
			}

			allowed, err := ProjectionTargetAllowed(source, rack, &rack.Spec.Nodes[0], node)

			require.NoError(t, err)
			require.Equal(t, tt.want, allowed)
		})
	}
}
