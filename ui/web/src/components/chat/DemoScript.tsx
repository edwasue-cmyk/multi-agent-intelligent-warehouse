import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  CheckCircle as CheckIcon,
  RadioButtonUnchecked as UncheckedIcon,
  Assignment as AssignmentIcon,
  Inventory as InventoryIcon,
  Security as SecurityIcon,
} from '@mui/icons-material';

interface DemoScriptProps {
  onScenarioSelect: (scenario: string) => void;
}

const DemoScript: React.FC<DemoScriptProps> = ({ onScenarioSelect }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);

  const demoFlows = [
    {
      id: 'work_order_flow',
      title: 'Work Order → Wave → Equipment Dispatch',
      description: 'Operations + EAO Agent collaboration',
      icon: <AssignmentIcon />,
      steps: [
        {
          label: 'User Request',
          description: 'User: "Create a wave for Orders 1001–1010 in Zone A and dispatch a forklift."',
          expected: 'Route → Operations (0.88)',
        },
        {
          label: 'Data Retrieval',
          description: 'Retrieval → top-12→6; SQL for order lines (38ms)',
          expected: 'Evidence: Order lines, zone layout',
        },
        {
          label: 'Proposed Actions',
          description: 'generate_pick_wave (params shown, SLA gain + travel time)',
          expected: 'Action card with Approve/Reject',
        },
        {
          label: 'Equipment Assignment',
          description: 'assign_equipment(asset_id=FL-07, zone=A)',
          expected: 'Equipment assignment card',
        },
        {
          label: 'User Approval',
          description: 'User clicks Approve (Tier 1)',
          expected: 'Success toast + audit IDs',
        },
        {
          label: 'Results',
          description: 'WMS wave id + equipment assignment',
          expected: 'action_result with links',
        },
      ],
    },
    {
      id: 'equipment_status_flow',
      title: 'Equipment Status Check & Assignment',
      description: 'Equipment/SQL path with asset management',
      icon: <InventoryIcon />,
      steps: [
        {
          label: 'User Request',
          description: 'User: "Show me the status of all forklifts and their availability"',
          expected: 'Route: Equipment (0.95)',
        },
        {
          label: 'Equipment Query',
          description: 'Execute equipment status query with filters',
          expected: 'SQL inspector shows query + results',
        },
        {
          label: 'Results Table',
          description: 'Table card: Equipment status breakdown',
          expected: 'Structured data table with asset details',
        },
        {
          label: 'Assignment Proposal',
          description: 'Proposed action assign_equipment for available forklifts',
          expected: 'Action card with equipment parameters',
        },
        {
          label: 'Approval',
          description: 'Approve → action_result with assignment details',
          expected: 'Success confirmation with asset ID',
        },
      ],
    },
    {
      id: 'safety_incident_flow',
      title: 'Safety Incident Response',
      description: 'Safety + EAO Agent collaboration',
      icon: <SecurityIcon />,
      steps: [
        {
          label: 'Incident Report',
          description: 'User: "Machine over-temp event at Dock D2."',
          expected: 'Route: Safety (0.95)',
        },
        {
          label: 'Proposed Actions',
          description: 'broadcast_alert(zone=D2), lockout_tagout_request(asset_id=DL-4), start_checklist(forklift_pre_op)',
          expected: 'Multiple action cards',
        },
        {
          label: 'User Decisions',
          description: 'Approve first two; reject third',
          expected: 'Selective approval/rejection',
        },
        {
          label: 'Evidence Panel',
          description: 'Policy spans + telemetry rows (Timescale)',
          expected: 'Evidence with source links',
        },
        {
          label: 'Summary',
          description: 'Final message summarizes actions, links to tickets, next steps',
          expected: 'Comprehensive action summary',
        },
      ],
    },
  ];

  const handleStepComplete = (stepIndex: number) => {
    if (!completedSteps.includes(stepIndex)) {
      setCompletedSteps([...completedSteps, stepIndex]);
    }
  };

  const handleFlowStart = (flowId: string) => {
    onScenarioSelect(flowId);
    setActiveStep(0);
    setCompletedSteps([]);
  };

  const getStepIcon = (stepIndex: number) => {
    if (completedSteps.includes(stepIndex)) {
      return <CheckIcon sx={{ color: '#76B900' }} />;
    }
    return <UncheckedIcon sx={{ color: '#666666' }} />;
  };

  return (
    <Box sx={{ p: 2, backgroundColor: '#111111', height: '100%', overflow: 'auto' }}>
      <Typography variant="h6" sx={{ color: '#ffffff', mb: 3, textAlign: 'center' }}>
        Demo Scripts
      </Typography>

      {demoFlows.map((flow, flowIndex) => (
        <Card key={flow.id} sx={{ mb: 3, backgroundColor: '#1a1a1a', border: '1px solid #333333' }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Box sx={{ color: '#76B900', mr: 2 }}>
                {flow.icon}
              </Box>
              <Box sx={{ flex: 1 }}>
                <Typography variant="h6" sx={{ color: '#ffffff', fontSize: '16px' }}>
                  {flow.title}
                </Typography>
                <Typography variant="body2" sx={{ color: '#666666' }}>
                  {flow.description}
                </Typography>
              </Box>
              <Button
                variant="contained"
                startIcon={<PlayIcon />}
                onClick={() => handleFlowStart(flow.id)}
                sx={{
                  backgroundColor: '#76B900',
                  '&:hover': { backgroundColor: '#5a8f00' },
                }}
              >
                Start
              </Button>
            </Box>

            <Stepper activeStep={activeStep} orientation="vertical">
              {flow.steps.map((step, stepIndex) => (
                <Step key={stepIndex}>
                  <StepLabel
                    StepIconComponent={() => getStepIcon(stepIndex)}
                    sx={{
                      '& .MuiStepLabel-label': {
                        color: '#ffffff',
                        fontSize: '14px',
                      },
                    }}
                  >
                    {step.label}
                  </StepLabel>
                  <StepContent>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" sx={{ color: '#cccccc', mb: 1 }}>
                        {step.description}
                      </Typography>
                      <Chip
                        label={step.expected}
                        size="small"
                        sx={{
                          backgroundColor: '#333333',
                          color: '#76B900',
                          fontSize: '10px',
                        }}
                      />
                    </Box>
                    <Button
                      size="small"
                      onClick={() => handleStepComplete(stepIndex)}
                      disabled={completedSteps.includes(stepIndex)}
                      sx={{
                        color: '#76B900',
                        '&:disabled': {
                          color: '#666666',
                        },
                      }}
                    >
                      {completedSteps.includes(stepIndex) ? 'Completed' : 'Mark Complete'}
                    </Button>
                  </StepContent>
                </Step>
              ))}
            </Stepper>
          </CardContent>
        </Card>
      ))}

      <Card sx={{ backgroundColor: '#1a1a1a', border: '1px solid #333333' }}>
        <CardContent>
          <Typography variant="h6" sx={{ color: '#ffffff', mb: 2, fontSize: '16px' }}>
            Demo Tips
          </Typography>
          <List dense>
            <ListItem>
              <ListItemIcon>
                <CheckIcon sx={{ color: '#76B900', fontSize: '16px' }} />
              </ListItemIcon>
              <ListItemText
                primary="Choose WH-01, Role Manager"
                primaryTypographyProps={{ fontSize: '12px', color: '#ffffff' }}
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <CheckIcon sx={{ color: '#76B900', fontSize: '16px' }} />
              </ListItemIcon>
              <ListItemText
                primary="Toggle Show Internals for detailed view"
                primaryTypographyProps={{ fontSize: '12px', color: '#ffffff' }}
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <CheckIcon sx={{ color: '#76B900', fontSize: '16px' }} />
              </ListItemIcon>
              <ListItemText
                primary="Check Evidence & Tool timeline in right panel"
                primaryTypographyProps={{ fontSize: '12px', color: '#ffffff' }}
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <CheckIcon sx={{ color: '#76B900', fontSize: '16px' }} />
              </ListItemIcon>
              <ListItemText
                primary="Use quick replies for clarifying questions"
                primaryTypographyProps={{ fontSize: '12px', color: '#ffffff' }}
              />
            </ListItem>
          </List>
        </CardContent>
      </Card>
    </Box>
  );
};

export default DemoScript;
