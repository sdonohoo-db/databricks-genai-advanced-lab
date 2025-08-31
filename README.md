# Databricks GenAI Advanced Lab

A practical 2-day hands-on workshop for building enterprise-level AI agents using Databricks platform capabilities.

## Overview

This lab guides you through creating intelligent AI agents from data exploration to production deployment, covering vector search, function calling, evaluation, monitoring, and user interfaces.

## Prerequisites

- Databricks workspace access
- Basic Python knowledge

## Lab Structure

### Day 1: Foundational Agent Building

**Setup & Data Exploration**
- `00_setup/` - Environment configuration and data preparation
- `01_explore_data/` - Data analysis and preprocessing techniques

**Vector Search & Tools**
- `02_create_vector_search_index/` - Building semantic search capabilities
- `03_create_tools/` - Creating custom function tools for agents

**Agent Development**
- `04_create_agent_with_vsi_and_tools/` - Core agent with vector search integration
- `05_eval_agent_and_deploy/` - Agent evaluation and deployment strategies

**Production Readiness**
- `06_setup_sme_review/` - Subject matter expert review workflows
- `07_monitor_agent_in_production/` - Production monitoring and observability

**User Interfaces**
- `08_create_chatbot_app/` - Web-based chatbot application

### Day 2: New & Advanced Features

**Multi-Agent Development**
- `09_create_genie_space (UI)/` - Databricks Genie space integration
- `10_create_multi_agent_with_tools_and_genie (ChatAgent)/` - Multi-agent systems

**Advanced Architectures**
- `11_create_agent_with_mcp/` - Model Control Protocol integration
- `12_create_agent_bricks (UI)/` - Agent Bricks UI components
- `13_prompt_optimization/` - Advanced prompt engineering techniques

## Key Learning Outcomes

- **Build Agent Systems**: Create single-agent and multi-agent architectures with vector search, function calling, and tool integration
- **Master LLMOps Pipeline**: Implement the complete lifecycle from development to production including evaluation, monitoring, deployment, and Databricks Apps for front-end applications  
- **Compare Agent Approaches**: Hands-on experience with different architectures (single-agent vs multi-agent vs MCP), prompt optimization, and quick deployment approaches like Agent Bricks and Genie spaces

## Data

The `data/` directory contains sample ecommerce datasets:
- **CSV files**: customer services, product documentation, policies, inventories
- **PDF files**: sample product manuals for hands-on document processing

## Getting Started

1. Run `00_setup/00_setup.py` to initialize your environment
2. Follow modules sequentially for best learning experience
3. Each module builds upon previous concepts

