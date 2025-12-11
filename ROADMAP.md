# Llama Nexus UI Roadmap

A focused development plan for UI refinements and new features.

---

## Phase 1: Core Experience Polish

**Goal:** Refine the most-used interfaces for a seamless daily workflow.

### Chat Interface
- [ ] Conversation branching (fork from any message)
- [x] Message editing with re-generation
- [x] Keyboard shortcuts (Cmd+L, Cmd+N, Cmd+S, etc.)
- [ ] Collapsible thinking/reasoning blocks
- [ ] Syntax highlighting themes selection

### Dashboard
- [ ] Customizable widget layout (drag-and-drop)
- [x] Quick actions bar (one-click navigation to key features)
- [ ] Recent conversations widget
- [ ] Resource usage sparklines

### Navigation
- [x] Collapsible sidebar sections
- [ ] Favorites/pinned pages
- [x] Command palette (Cmd+K) for quick navigation
- [x] Breadcrumb trail for deep pages

---

## Phase 2: Deployment Consolidation

**Goal:** Unify the four deployment pages into a cohesive experience.

### Unified Deploy Dashboard
- [ ] Single page with service cards (LLM, Embedding, STT, TTS)
- [ ] At-a-glance status for all services
- [ ] One-click start/stop for each service
- [ ] Resource allocation overview

### Service Configuration
- [ ] Shared configuration panel pattern
- [ ] Preset configurations (fast, balanced, quality)
- [ ] VRAM budget calculator across all services
- [ ] Dependency indicators (e.g., "Requires LLM for tool calls")

---

## Phase 3: Knowledge Management

**Goal:** Streamline document ingestion and RAG workflows.

### Documents Page
- [x] Drag-and-drop upload zone
- [ ] Batch upload with progress queue
- [ ] Document preview panel
- [ ] Processing status indicators (chunked, embedded, extracted)

### Knowledge Graph
- [ ] Mini-map for large graphs
- [ ] Node clustering by type
- [ ] Search-to-highlight in graph
- [ ] Export graph as image/JSON

### RAG Search
- [ ] Source highlighting in results
- [ ] Relevance score visualization
- [ ] Filter by document/date/type
- [ ] "Ask follow-up" on results

---

## Phase 4: Developer Tools

**Goal:** Improve iteration speed for prompt engineering and testing.

### Prompt Library
- [ ] Side-by-side prompt comparison
- [ ] Variable playground with live preview
- [ ] Usage analytics per prompt
- [ ] Import/export to JSON/YAML

### Workflows
- [ ] Visual node editor improvements
- [ ] Execution visualization (animated flow)
- [ ] Debug mode with step-through
- [ ] Template gallery

### Testing & Benchmark
- [ ] Test suite management
- [ ] Automated regression testing
- [ ] Historical benchmark charts
- [ ] A/B comparison reports

---

## Phase 5: Global Refinements

**Goal:** Consistent, polished experience across all pages.

### Design System
- [ ] Consistent card patterns
- [x] Unified loading states (PageSkeleton)
- [x] Empty state components
- [x] Toast notifications system

### Accessibility
- [ ] Keyboard navigation for all controls
- [ ] Screen reader improvements
- [ ] High contrast mode
- [ ] Reduced motion option

### Performance
- [ ] Lazy loading for heavy pages
- [ ] Virtualized lists for large datasets
- [ ] Optimistic UI updates
- [ ] Background data refresh

---

## Quick Wins

Immediate improvements with high impact:

| Item | Location | Effort | Status |
|------|----------|--------|--------|
| Loading skeletons | All pages | Low | Done |
| Toast notifications | Global | Low | Done |
| Error state improvements | All pages | Low | |
| Tooltip consistency | Navigation | Low | |
| Form validation feedback | Config pages | Medium | |
| Dark/light theme toggle | Header | Medium | |
| Export functionality | Chat, Documents | Medium | |

---

## Metrics

Track success through:

- Time to first interaction (page load)
- Task completion rate (deploy model, upload doc)
- Error recovery rate
- User session duration

---

*Last updated: December 2024*
