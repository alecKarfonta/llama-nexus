import React from 'react'
import {
  Grid,
  Button,
} from '@mui/material'
import {
  Add as AddIcon,
  History as HistoryIcon,
  RocketLaunch as DeployIcon,
  Route as RouteIcon,
} from '@mui/icons-material'
import { useNavigate } from 'react-router-dom'
import { ServiceStatusDisplay } from '@/components/monitoring/ServiceStatusDisplay'
import { LogViewer } from '@/components/monitoring/LogViewer'
import { TokenUsageTracker } from '@/components/monitoring/TokenUsageTracker'
import { RecentConversations, WorkflowRunway } from '@/components/dashboard'
import { PageShell, SectionCard, compactCollapsibleCardSx } from '@/components/common'
import {
  getRoutePath,
} from '@/config/navigation'
import { useTokenUsage } from '@/hooks/useMetrics'
import { accentColors } from '@/utils/theme'

const compactButtonSx = {
  height: 34,
  minHeight: 34,
  py: 0,
  px: 1.25,
  borderRadius: '8px',
  borderColor: 'divider',
  textTransform: 'none',
  fontWeight: 800,
} as const

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate()
  const chatPath = getRoutePath('chat')
  const deployPath = getRoutePath('deploy')
  const tokenUsageRange = '24h'
  const {
    data: tokenUsage = [],
    isLoading: isTokenUsageLoading,
    error: tokenUsageError,
  } = useTokenUsage(tokenUsageRange)
  const showTokenUsage = isTokenUsageLoading || Boolean(tokenUsageError) || tokenUsage.length > 0

  return (
    <PageShell sx={{ py: 1.25 }}>
      {/* [2026-10-16] | INFO | DASHBOARD_PAGE.WORKFLOW_RUNWAY | Guided-tour
          section reviving the navigation.tsx workflow/lifecycle data with
          live feature-usage chips. */}
      <Grid container spacing={1.25} sx={{ mb: 1.25 }}>
        <Grid item xs={12}>
          <SectionCard
            title="Workflow Runway"
            subtitle="The llama-nexus path from assets to operations"
            accentColor={accentColors.purple}
            icon={<RouteIcon />}
            collapsible
            defaultExpanded={false}
            sx={compactCollapsibleCardSx}
          >
            <WorkflowRunway hasTokenUsage={tokenUsage.length > 0} />
          </SectionCard>
        </Grid>
      </Grid>

      <Grid container spacing={1.25} sx={{ mb: 1.25 }}>
        <Grid item xs={12} lg={7}>
          <SectionCard
            title="Runtime Control"
            subtitle="Current service state and launch controls"
            accentColor={accentColors.success}
            icon={<DeployIcon />}
            action={
              <Button size="small" variant="outlined" onClick={() => navigate(deployPath)} sx={compactButtonSx}>
                Deploy
              </Button>
            }
          >
            <ServiceStatusDisplay />
          </SectionCard>
        </Grid>

        <Grid item xs={12} lg={5}>
          <SectionCard
            title="Recent Conversations"
            subtitle="Latest chat sessions"
            accentColor={accentColors.info}
            icon={<HistoryIcon />}
            action={
              <Button
                size="small"
                variant="outlined"
                startIcon={<AddIcon />}
                onClick={() => navigate(chatPath)}
                sx={compactButtonSx}
              >
                New
              </Button>
            }
          >
            <RecentConversations limit={3} showStats={false} showSearch={false} compact />
          </SectionCard>
        </Grid>
      </Grid>

      <Grid container spacing={1.25}>
        {showTokenUsage && (
          <Grid item xs={12} lg={5}>
            <SectionCard
              title="Token Usage"
              subtitle="Consumption and request accounting"
              accentColor={accentColors.purple}
            >
              <TokenUsageTracker compact timeRange={tokenUsageRange} />
            </SectionCard>
          </Grid>
        )}

        <Grid item xs={12} lg={showTokenUsage ? 7 : 12}>
          <SectionCard
            title="System Logs"
            accentColor={accentColors.warning}
            collapsible
            defaultExpanded={false}
            sx={compactCollapsibleCardSx}
          >
            <LogViewer maxLines={200} height={140} compact collapseEmpty />
          </SectionCard>
        </Grid>
      </Grid>
    </PageShell>
  )
}
