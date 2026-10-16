/**
 * WorkflowRunway - Revives the guided-tour navigation data
 * (dashboardWorkflowRunway / dashboardLifecycleStages) on the Dashboard page.
 *
 * Each workflow step shows title, short description, and a status chip that
 * flips to Complete when the matching feature has been used (approximated from
 * live service/model/conversation state; anything without a cheap probe stays
 * "Not started").
 */
import React, { useEffect, useState } from 'react'
import {
  Box,
  Button,
  Typography,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import { useNavigate } from 'react-router-dom'
import {
  dashboardLifecycleStages,
  dashboardWorkflowRunway,
} from '@/config/navigation'
import { StatusChip } from '@/components/common'
import { apiService } from '@/services/api'

// [2026-10-16] | INFO | WORKFLOW_RUNWAY.FEATURE_USAGE | Approximate
// "has the user used this feature yet" probes. One lazy fetch per mount.
interface FeatureUsage {
  models: boolean
  deploy: boolean
  chat: boolean
  knowledge: boolean
  finetuning: boolean
  operations: boolean
}

const INITIAL_USAGE: FeatureUsage = {
  models: false,
  deploy: false,
  chat: false,
  knowledge: false,
  finetuning: false,
  operations: false,
}

// Maps a navigation item id to the usage probe that proves the step is done.
const USAGE_KEY_BY_PAGE_ID: Record<string, keyof FeatureUsage> = {
  models: 'models',
  registry: 'models',
  deploy: 'deploy',
  chat: 'chat',
  search: 'knowledge',
  'knowledge-graph': 'knowledge',
  finetuning: 'finetuning',
  'finetuning-distillation': 'finetuning',
  'finetuning-evaluation': 'finetuning',
  benchmark: 'operations',
  'retrieval-analytics': 'operations',
  monitoring: 'operations',
}

const usageKeyForPageId = (id: string): keyof FeatureUsage | undefined => (
  USAGE_KEY_BY_PAGE_ID[id]
)

export interface WorkflowRunwayProps {
  /** Extra proof signal from the host page (e.g. token usage history exists). */
  hasTokenUsage?: boolean
}

export const WorkflowRunway: React.FC<WorkflowRunwayProps> = ({ hasTokenUsage = false }) => {
  const navigate = useNavigate()
  const [usage, setUsage] = useState<FeatureUsage>(INITIAL_USAGE)

  // [2026-10-16] | INFO | WORKFLOW_RUNWAY.USE_EFFECT_PROBES | Best-effort
  // probes: service running, any local model file, any conversation, any
  // ingested GraphRAG document. Failures keep the chip at "Not started".
  useEffect(() => {
    let cancelled = false
    const probe = async () => {
      const next: FeatureUsage = { ...INITIAL_USAGE }
      try {
        const status = await apiService.getServiceStatus()
        next.deploy = status.health === 'healthy' && status.modelLoaded
      } catch { /* keep false */ }
      try {
        const models = await apiService.getModels()
        next.models = models.length > 0
      } catch { /* keep false */ }
      try {
        const { conversations } = await apiService.listConversations({ limit: 1 })
        next.chat = conversations.length > 0
      } catch { /* keep false */ }
      try {
        const docs = await apiService.listGraphRAGDocuments()
        next.knowledge = docs.total_documents > 0
      } catch { /* keep false */ }
      next.operations = hasTokenUsage
      if (!cancelled) setUsage(next)
    }
    void probe()
    return () => { cancelled = true }
  }, [hasTokenUsage])

  const chipForPageId = (pageId: string) => {
    const key = usageKeyForPageId(pageId)
    const complete = key !== undefined && usage[key]
    return complete
      ? <StatusChip status="completed" label="Complete" height={20} />
      : <StatusChip status="pending" label="Not started" height={20} />
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* [2026-10-16] | INFO | WORKFLOW_RUNWAY.RENDER | Workflow runway: the
          five-phase guided path from assets to operations. */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {dashboardWorkflowRunway.map((step) => (
          <Box
            key={step.id}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1.25,
              p: 1,
              borderRadius: 1,
              border: '1px solid',
              borderColor: 'divider',
              bgcolor: alpha(step.color, 0.04),
            }}
          >
            <step.icon style={{ fontSize: 20, color: step.color, flexShrink: 0 }} />
            <Box sx={{ flexGrow: 1, minWidth: 0 }}>
              <Typography variant="body2" fontWeight={600} noWrap>
                {step.phase} · {step.title}
              </Typography>
              <Typography variant="caption" color="text.secondary" component="span">
                {step.summary}
              </Typography>
            </Box>
            {chipForPageId(step.id)}
            <Button
              size="small"
              variant="outlined"
              sx={{ flexShrink: 0, textTransform: 'none' }}
              onClick={() => navigate(step.primaryAction.path)}
            >
              {step.primaryAction.label}
            </Button>
          </Box>
        ))}
      </Box>

      {/* [2026-10-16] | INFO | WORKFLOW_RUNWAY.RENDER_LIFECYCLE | Lifecycle
          stages: the same journey expressed as linked feature surfaces. */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
        {dashboardLifecycleStages.map((stage) => (
          <Box
            key={stage.id}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              flexWrap: 'wrap',
              py: 0.5,
            }}
          >
            <stage.icon style={{ fontSize: 16, color: stage.color, flexShrink: 0 }} />
            <Typography variant="caption" fontWeight={600} sx={{ mr: 0.5 }}>
              {stage.title}
            </Typography>
            {stage.links.map((link) => (
              <Button
                key={`${stage.id}-${link.id}`}
                size="small"
                sx={{ minHeight: 22, px: 0.75, fontSize: '0.7rem', textTransform: 'none' }}
                onClick={() => navigate(link.path)}
              >
                {link.label}
              </Button>
            ))}
            {chipForPageId(stage.id)}
          </Box>
        ))}
      </Box>
    </Box>
  )
}

export default WorkflowRunway
