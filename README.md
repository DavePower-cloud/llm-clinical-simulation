**LLM Clinical Simulation**

Multi-agent large language model framework for simulating clinical conversations in healthcare training environments, with automated evaluation using LLM-based judging.

This repository contains code for generating and evaluating simulated clinical conversations between healthcare roles using large language models. The system supports role-locked agents, conversation orchestration, and automated scoring of conversation quality using an AI-as-judge approach.

The framework was developed as part of research into the use of generative AI to model clinical communication and training scenarios.

**Overview**

Clinical communication is central to safe patient care, yet studying communication behaviour is difficult due to the cost and complexity of real-world observation and simulation.

This project explores whether large language models can simulate realistic clinical conversations between healthcare professionals and patients.

The framework uses multiple LLM agents assigned specific clinical roles (e.g., nurse, doctor, patient). These agents interact within a controlled conversation loop, producing simulated dialogues representing clinical scenarios.

Generated conversations can then be evaluated automatically using a LLM-based judge model that scores conversation realism, role fidelity, communication clarity, and educational usefulness.

**Key Features**

Multi-agent LLM clinical conversation simulation

Role-locked agents (e.g., nurse, doctor, patient)

Support for multiple model providers (OpenAI, Anthropic, Google)

Structured prompt templates for each clinical role

Conversation persistence in JSON / JSONL format

Batch generation of simulated clinical conversations

Automated evaluation using AI-as-judge

Support for reproducible research experiments
