# Architecture Documentation Standard

## Purpose and Scope

**Purpose**: Create comprehensive architecture documentation optimized for subsequent implementation planning
**Target Audience**: Software architects and technical leads creating system architecture documents and architecture agents creating implementation plans from the architecture document
**Output Goal**: Architecture documentation from which we can derive an implementation plan that enables rapid, systematic development

---

## Document Structure Requirements

### 1. Executive Summary
- **Clear system purpose and business value**
- **Technology stack overview** (FastAPI, Python 3.13+, PostgreSQL, Redis, HTMX)
- **Core architectural principles** (function-based design, dependency injection, performance-first)
- **Success metrics and performance targets**

### 2. System Overview
- **Visual ASCII diagrams** showing component relationships
- **Data flow patterns** and service communication
- **Integration points** with existing systems
- **Security and authentication** patterns

### 3. Technical Architecture Details

#### Core Components (Required Sections):
- **Domain Layer**: Pydantic models, validators, protocols
- **Service Layer**: Business logic functions with FastAPI DI
- **Repository Layer**: Data access with function-based pattern
- **API Layer**: Route handlers and request/response models
- **Template Layer**: Jinja2 templates and rendering functions

#### Architecture Patterns:
- **Function-Based Design**: No classes except Pydantic models, database models, global storage
- **Dependency Injection**: FastAPI-compatible patterns throughout
- **Plugin Architecture**: Extensible hook systems where applicable
- **Notification Integration**: Unified error/success feedback
- **Performance Optimization**: Built-in metrics and monitoring

### 4. Implementation-Ready Specifications

#### File Structure:
Provide complete directory structure with:
```
backend/services/[module]/
├── domain/          # Models, validators, protocols
├── services/        # Business logic functions
├── repositories/    # Data access functions
├── routes/          # FastAPI route handlers
├── templates/       # Jinja2 templates
├── tests/           # pytest-BDD tests
```

#### Database Schema:
- **Complete SQL schemas** with relationships
- **Index specifications** for performance
- **Migration considerations**
- **Data validation rules**

#### API Endpoints:
- **Complete endpoint specifications**
- **Request/response models** with examples
- **Authentication requirements**
- **Rate limiting and caching** strategies

### 5. Performance & Quality Specifications

#### Performance Targets:
- **Response time requirements** (e.g., <100ms for table operations)
- **Concurrent user targets** (e.g., 100+ users)
- **Memory usage limits**
- **Database query performance**

#### Testing Requirements:
- **pytest-BDD with Gherkin** scenarios
- **Test coverage targets** (>90%)
- **Performance benchmarks**
- **Integration testing approach**

### 6. Implementation Guidance

#### Code Examples:
- **Function-based service implementations**
- **FastAPI dependency injection patterns**
- **Pydantic model definitions**
- **Template rendering examples**

#### Configuration:
- **Environment variables**
- **Docker integration**
- **Redis caching setup**
- **Database connection management**

---

## Documentation Quality Standards

### Technical Precision:
- **Specific metrics and measurements**
- **Copy-pasteable code examples**
- **Production-ready configurations**
- **Error handling patterns**

### Implementation Alignment:
- **Clear function signatures**
- **Dependency relationships**
- **Integration patterns**
- **Deployment considerations**

### Architecture Compliance:
- **Function-based design emphasis**
- **Performance optimization built-in**
- **Security considerations**
- **Scalability patterns**

---

## Output Format Requirements

Structure your architecture document with:

1. **Clear section headers and navigation**
2. **ASCII diagrams for complex relationships**
3. **Code examples with syntax highlighting**
4. **Performance metrics and targets**
5. **Implementation checklists**
6. **Integration patterns**
7. **Security considerations**
8. **Deployment guidelines**

---

## Success Criteria

Your architecture document should enable:

- **Rapid implementation** with minimal ambiguity
- **Systematic development** following clear patterns
- **Maximum technical precision** in specifications
- **Production-ready deployment** guidance
- **Comprehensive testing** strategies
- **Performance optimization** from day one

---

## Key Architectural Principles to Emphasize

### Function-Based Design
- All business logic as pure functions
- Classes only for: Pydantic models, database models, global storage, exceptions, enums
- Clear separation of concerns between layers

### FastAPI Integration
- Dependency injection for all service functions
- Comprehensive request/response validation
- Built-in documentation generation
- Performance optimization through async patterns

### Performance First
- Response time targets built into design
- Caching strategies at every layer
- Database query optimization
- Memory efficiency considerations

### Security by Design
- Input validation at all boundaries
- Authentication and authorization patterns
- SQL injection prevention
- XSS protection in templates

### Testing Excellence
- pytest-BDD with Gherkin scenarios
- Comprehensive test coverage
- Performance benchmarks
- Integration testing strategies

---

## Template Structure

Use this template structure for consistent architecture documentation:

```markdown
# [System Name] Architecture

## Executive Summary
[Purpose, value, stack, principles, metrics]

## System Overview
[Diagrams, data flow, integrations, security]

## Technical Architecture
### Domain Layer
[Models, validators, protocols]

### Service Layer
[Business logic functions]

### Repository Layer
[Data access patterns]

### API Layer
[Routes, authentication]

### Template Layer
[UI rendering]

## Implementation Specifications
### File Structure
[Complete directory layout]

### Database Schema
[SQL schemas with indexes]

### API Endpoints
[Complete endpoint specs]

## Performance & Quality
### Performance Targets
[Specific metrics]

### Testing Requirements
[Coverage and strategies]

## Implementation Guidance
### Code Examples
[Function-based patterns]

### Configuration
[Environment setup]

## Deployment
[Production considerations]
```

---

## Architecture Document Checklist

Before finalizing any architecture document, ensure it includes:

### ✅ Executive Summary
- [ ] Clear system purpose and business value statement
- [ ] Complete technology stack specification
- [ ] Core architectural principles defined
- [ ] Quantifiable success metrics and performance targets

### ✅ System Overview
- [ ] ASCII diagrams showing component relationships
- [ ] Data flow patterns documented
- [ ] Integration points with existing systems mapped
- [ ] Security and authentication patterns defined

### ✅ Technical Architecture
- [ ] Domain layer specifications (models, validators, protocols)
- [ ] Service layer business logic functions
- [ ] Repository layer data access patterns
- [ ] API layer route handlers and models
- [ ] Template layer rendering functions

### ✅ Implementation Specifications
- [ ] Complete file structure provided
- [ ] Database schemas with relationships and indexes
- [ ] API endpoints with request/response examples
- [ ] Authentication and caching strategies

### ✅ Performance & Quality
- [ ] Specific response time requirements
- [ ] Concurrent user targets
- [ ] Memory usage limits
- [ ] Testing coverage targets (>90%)

### ✅ Implementation Guidance
- [ ] Function-based code examples
- [ ] FastAPI dependency injection patterns
- [ ] Production-ready configuration examples
- [ ] Error handling patterns

### ✅ Deployment Considerations
- [ ] Environment variable specifications
- [ ] Docker integration guidance
- [ ] Database migration strategies
- [ ] Monitoring and logging setup

---

## Quality Assurance Guidelines

### Code Examples Must:
- Be copy-pasteable and immediately functional
- Follow function-based design principles
- Include proper type hints and validation
- Demonstrate FastAPI dependency injection
- Show error handling patterns

### Performance Specifications Must:
- Include specific numeric targets (e.g., "<100ms response time")
- Address concurrent user scenarios
- Specify database query optimization
- Include caching strategies

### Testing Requirements Must:
- Specify pytest-BDD with Gherkin scenarios
- Define coverage targets (minimum 90%)
- Include performance benchmarks
- Address integration testing approaches

### Security Considerations Must:
- Address input validation at all boundaries
- Define authentication and authorization patterns
- Include SQL injection prevention measures
- Specify XSS protection in templates

Focus on creating documentation that enables rapid, systematic implementation with minimal ambiguity and maximum technical precision.