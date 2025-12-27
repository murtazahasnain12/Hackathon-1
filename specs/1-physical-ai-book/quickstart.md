# Quickstart Guide: Physical AI & Humanoid Robotics Book

## Getting Started

This guide will help you set up the development environment for the Physical AI & Humanoid Robotics book project.

### Prerequisites

- Node.js 18+ installed
- Git for version control
- Access to academic databases for research (IEEE Xplore, ACM Digital Library, arXiv)
- Text editor or IDE of choice

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the local development server**
   ```bash
   npm start
   ```

4. **Open your browser to** `http://localhost:3000` to view the book

### Project Structure Overview

The book content is organized in the `docs/` directory following the layered architecture:

- `docs/intro.md` - Introduction to Physical AI & Embodied Intelligence
- `docs/chapter-1/` - ROS 2 Foundations for Humanoids
- `docs/chapter-2/` - Digital Twins & Physics Simulation
- `docs/chapter-3/` - AI Perception & Learning with NVIDIA Isaac
- `docs/chapter-4/` - Vision-Language-Action Pipelines
- `docs/chapter-5/` - Autonomous Humanoid Capstone Architecture
- `docs/architecture/` - System architecture diagrams and explanations
- `docs/hardware/` - Infrastructure and lab setup information
- `docs/references/` - Comprehensive citations and references

### Writing Content

1. **Create a new chapter** in the appropriate directory
2. **Follow the content template** with learning objectives, content, and references
3. **Include proper citations** in APA format
4. **Ensure content meets readability standards** (Flesch-Kincaid grade 10-12)
5. **Add diagrams and code examples** as needed

### Adding References

When adding content, ensure you include proper citations:

```markdown
According to recent research [1], Physical AI represents a significant advancement in robotics.

## References

1. [Author et al., 2025]. "Title of the paper." *Journal Name*. APA format.
```

### Quality Assurance

Before committing changes, ensure:

- Content meets academic standards (40%+ primary/peer-reviewed sources)
- Writing is at grade 10-12 level
- No plagiarism (use plagiarism detection tools)
- All citations follow APA format
- Architecture diagrams accurately represent the layered system

### Building for Production

To build the static site for deployment:

```bash
npm run build
```

The output will be in the `build/` directory and can be deployed to GitHub Pages.

### Deployment

The site is automatically deployed to GitHub Pages when changes are merged to the main branch. Ensure all content passes quality checks before merging.