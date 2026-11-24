# Karpenter

Karpenter is a Kubernetes node provisioning project that automatically
launches and terminates nodes to meet the needs of unscheduled pods.

## Installation

Karpenter is installed via Helm. It requires an IAM role and an EC2
instance profile to launch nodes on AWS.

## Provisioners

A NodePool defines constraints and requirements used by Karpenter when
provisioning nodes, such as instance types, zones, and capacity type.
