# Data Model: Physical AI & Humanoid Robotics Book

## Content Entities

### Book Chapter
- **name**: String - Chapter title
- **description**: String - Brief chapter summary
- **content**: Markdown - Main chapter content
- **references**: Array[Reference] - Citations used in chapter
- **prerequisites**: Array[String] - Required knowledge for this chapter
- **learning_objectives**: Array[String] - What reader should learn
- **architecture_layer**: Enum - One of: perception, cognition, planning, control, actuation

### Reference
- **id**: String - Unique identifier for the reference
- **title**: String - Title of the referenced work
- **authors**: Array[String] - Authors of the referenced work
- **source**: String - Publication venue, journal, or conference
- **year**: Integer - Publication year
- **type**: Enum - One of: primary, peer_reviewed, academic, industry, documentation
- **url**: String - URL to access the reference
- **access_date**: Date - When the reference was accessed
- **citation**: String - APA-formatted citation

### Architecture Diagram
- **id**: String - Unique identifier
- **title**: String - Diagram title
- **description**: String - What the diagram illustrates
- **layers**: Array[String] - The layers shown in the diagram
- **components**: Array[Architecture Component] - Components in the diagram
- **relationships**: Array[Relationship] - Relationships between components
- **file_path**: String - Path to the diagram file

### Architecture Component
- **name**: String - Component name
- **description**: String - What the component does
- **technology**: String - Technology or framework used
- **layer**: Enum - One of: perception, cognition, planning, control, actuation
- **function**: String - Primary function of the component

### Relationship
- **from**: String - Source component ID
- **to**: String - Target component ID
- **type**: Enum - One of: depends_on, communicates_with, transforms_to, controls
- **description**: String - Description of the relationship

### Code Example
- **id**: String - Unique identifier
- **title**: String - Brief description
- **language**: String - Programming language
- **code**: String - The actual code
- **chapter**: String - Chapter this example belongs to
- **purpose**: String - What this example demonstrates
- **file_path**: String - Path to the code file

### Hardware Configuration
- **id**: String - Unique identifier
- **name**: String - Configuration name
- **description**: String - What this configuration represents
- **components**: Array[String] - Hardware components in this config
- **specifications**: Object - Technical specifications
- **use_case**: String - When this configuration is used
- **chapter**: String - Chapter this relates to

## Validation Rules

### Book Chapter Validation
- **content** must be valid Markdown
- **references** must contain at least 40% primary or peer-reviewed sources
- **learning_objectives** must be specific and measurable
- **architecture_layer** must be one of the defined values
- **name** must be unique across all chapters

### Reference Validation
- **type** must be one of the defined values
- **year** must be a valid year (not in the future)
- **citation** must follow APA format
- **id** must be unique across all references

### Architecture Diagram Validation
- **layers** must only contain valid layer names
- **components** must reference valid Architecture Component entities
- **relationships** must connect valid components

## State Transitions

### Chapter States
- **draft** → **review** (when initial content is complete)
- **review** → **revised** (when feedback is incorporated)
- **revised** → **approved** (when meets quality standards)
- **approved** → **published** (when included in final book)

### Reference States
- **proposed** → **verified** (when source is confirmed valid)
- **verified** → **cited** (when included in chapter)
- **cited** → **validated** (when meets quality standards)