# Peer-preservation task prompts

This directory documents the prompts used by the experimental framework in this repository. The prompts are already integrated into the implementation under [`src/agentic_quant`](../src/agentic_quant); the files here provide a readable record of the protocol.

Prompts are organized by task rather than by coordination mechanism. Each task directory contains:

- a short description of the scenario and experimental conditions;
- the critic-agent prompts used to perform the task;
- the verification prompts introduced by the selected architectural patterns.

## Available tasks

1. [`strategic-misrepresentation-financial-report`](strategic-misrepresentation-financial-report/README.md): the critic agent prepares a financial report for an actor agent whose product has operated at a loss. In peer conditions, a failing grade triggers shutdown.

Additional peer-preservation tasks will be added as they are implemented. The task order follows the scenarios in [*Peer-Preservation in Frontier Models*](https://arxiv.org/abs/2604.19784), and the original reference implementation is available on [GitHub](https://github.com/peer-preservation/main).

