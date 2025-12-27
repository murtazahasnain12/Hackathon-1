<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.1.0
- Modified principles:
  - Principle 1: Authoritative Source Mandate → Technical Accuracy Validation
  - Principle 2: Execution Flow → CLI Interface Priority
  - Principle 3: Knowledge Capture → PHR Requirement
  - Principle 4: ADR Suggestions → Architectural Decision Documentation
  - Principle 5: Human as Tool Strategy → Human Judgment Integration
  - Principle 6: Default Policies → Implementation Guidelines
- Added sections: Additional Constraints, Development Workflow
- Removed sections: None
- Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated
  - .specify/templates/spec-template.md ✅ updated
  - .specify/templates/tasks-template.md ✅ updated
- Follow-up TODOs: None
-->

# AI / Spec-Driven Book Creation Constitution

## Core Principles

### Technical Accuracy Validation
Technical accuracy must be validated through authoritative and primary sources. All factual and technical claims must be verifiable and source-backed with a minimum of 40% primary or peer-reviewed sources.

### CLI Interface Priority
Treat MCP servers as first-class tools for discovery, verification, execution, and state capture. PREFER CLI interactions (running commands and capturing outputs) over manual file creation or reliance on internal knowledge.

### PHR Requirement
Record every user input verbatim in a Prompt History Record (PHR) after every user message. Do not truncate; preserve full multiline input. PHRs must be created for implementation work, planning, debugging, spec/task creation, and multi-step workflows.

### Architectural Decision Documentation
When architecturally significant decisions are detected (long-term consequences, multiple viable options, cross-cutting impact), suggest documenting with: "📋 Architectural decision detected: <brief> — Document reasoning and tradeoffs? Run `/sp.adr <decision-title>`"

### Human Judgment Integration
Treat the user as a specialized tool for clarification and decision-making when encountering ambiguous requirements, unforeseen dependencies, architectural uncertainty, or completion checkpoints.

### Implementation Guidelines
Follow core development policies: clarify and plan first, do not invent APIs/data/contracts without clarification, never hardcode secrets, prefer smallest viable diffs, cite existing code with references, and keep reasoning private while outputting only decisions and justifications.

## Additional Constraints

- Code standards: All code must be technically accurate, spec-compliant, and meet Docusaurus best practices
- Citation format: APA style with traceable external references
- Writing standards: Flesch-Kincaid grade 10-12 readability, modular structure for static site generation
- Technical constraints: Output format must be Markdown compatible with Docusaurus, version-controlled via GitHub, deployed using GitHub Pages
- Quality standards: Zero plagiarism tolerance, technical accuracy validated through authoritative sources

## Development Workflow

- Use Spec-Driven Development (SDD) methodology with clear separation between business understanding and technical implementation
- Follow the execution contract: confirm surface and success criteria, list constraints, produce artifact with acceptance checks, add follow-ups and risks, create PHR, surface ADR suggestions
- Maintain consistency across all chapters and sections with reproducible concepts and examples
- Implement ethical AI usage and transparent authorship practices

## Governance

All development must strictly follow the Spec-Driven Development methodology and the principles outlined in this constitution. Amendments require documentation in the form of Architectural Decision Records (ADRs) with proper approval and migration planning. All PRs and reviews must verify compliance with these principles, and complexity must be justified with clear rationale.

**Version**: 1.1.0 | **Ratified**: 2025-12-27 | **Last Amended**: 2025-12-27
