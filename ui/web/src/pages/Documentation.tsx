import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Link,
  Alert,
  AlertTitle,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import {
  ExpandMore as ExpandMoreIcon,
  Code as CodeIcon,
  Architecture as ArchitectureIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Cloud as CloudIcon,
  BugReport as BugReportIcon,
  Rocket as RocketIcon,
  School as SchoolIcon,
  GitHub as GitHubIcon,
  Article as ArticleIcon,
  Build as BuildIcon,
  Api as ApiIcon,
  Dashboard as DashboardIcon,
} from '@mui/icons-material';

const Documentation: React.FC = () => {
  const navigate = useNavigate();
  const [expandedSection, setExpandedSection] = useState<string | false>('overview');

  const handleChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedSection(isExpanded ? panel : false);
  };

  const quickStartSteps = [
    {
      step: 1,
      title: "Environment Setup",
      description: "Set up Python virtual environment and install dependencies",
      code: "python -m venv env && source env/bin/activate && pip install -r requirements.txt"
    },
    {
      step: 2,
      title: "Database Configuration",
      description: "Configure PostgreSQL/TimescaleDB and Milvus connections",
      code: "cp .env.example .env && # Edit database credentials"
    },
    {
      step: 3,
      title: "NVIDIA NIMs Setup",
      description: "Configure NVIDIA API keys for LLM and embeddings",
      code: "export NVIDIA_API_KEY=your_api_key"
    },
    {
      step: 4,
      title: "Start Services",
      description: "Launch the application stack",
      code: "docker-compose up -d && python -m uvicorn chain_server.app:app --reload"
    }
  ];

  const architectureComponents = [
    {
      name: "Multi-Agent System",
      description: "Planner/Router + Specialized Agents (Equipment, Operations, Safety)",
      status: "✅ Production Ready",
      icon: <ArchitectureIcon />
    },
    {
      name: "NVIDIA NIMs Integration",
      description: "Llama 3.1 70B + NV-EmbedQA-E5-v5 embeddings",
      status: "✅ Fully Operational",
      icon: <CloudIcon />
    },
    {
      name: "MCP Framework",
      description: "Model Context Protocol with dynamic tool discovery",
      status: "✅ Phase 3 Complete",
      icon: <BuildIcon />
    },
    {
      name: "Hybrid RAG System",
      description: "PostgreSQL/TimescaleDB + Milvus vector search",
      status: "✅ Optimized",
      icon: <StorageIcon />
    },
    {
      name: "Advanced Reasoning",
      description: "5 reasoning types with confidence scoring",
      status: "✅ Implemented",
      icon: <SpeedIcon />
    },
    {
      name: "Security & RBAC",
      description: "JWT/OAuth2 with 5 user roles",
      status: "✅ Production Ready",
      icon: <SecurityIcon />
    }
  ];

  const apiEndpoints = [
    {
      category: "Core Chat",
      endpoints: [
        { method: "POST", path: "/api/v1/chat", description: "Main chat interface with agent routing" },
        { method: "GET", path: "/api/v1/health", description: "System health check" },
        { method: "GET", path: "/api/v1/metrics", description: "Prometheus metrics" }
      ]
    },
    {
      category: "Agent Operations",
      endpoints: [
        { method: "GET", path: "/api/v1/equipment/assignments", description: "Equipment assignments" },
        { method: "POST", path: "/api/v1/operations/waves", description: "Create pick waves" },
        { method: "POST", path: "/api/v1/safety/incidents", description: "Log safety incidents" }
      ]
    },
    {
      category: "MCP Framework",
      endpoints: [
        { method: "GET", path: "/api/v1/mcp/tools", description: "Discover available tools" },
        { method: "GET", path: "/api/v1/mcp/status", description: "MCP system status" },
        { method: "POST", path: "/api/v1/mcp/execute", description: "Execute MCP tools" }
      ]
    },
    {
      category: "Reasoning Engine",
      endpoints: [
        { method: "POST", path: "/api/v1/reasoning/analyze", description: "Deep reasoning analysis" },
        { method: "GET", path: "/api/v1/reasoning/types", description: "Available reasoning types" },
        { method: "POST", path: "/api/v1/reasoning/chat-with-reasoning", description: "Chat with reasoning" }
      ]
    }
  ];

  const toolsOverview = [
    {
      agent: "Equipment & Asset Operations",
      count: 8,
      tools: ["check_stock", "reserve_inventory", "create_replenishment_task", "generate_purchase_requisition", "adjust_reorder_point", "recommend_reslotting", "start_cycle_count", "investigate_discrepancy"]
    },
    {
      agent: "Operations Coordination",
      count: 8,
      tools: ["assign_tasks", "rebalance_workload", "generate_pick_wave", "optimize_pick_paths", "manage_shift_schedule", "dock_scheduling", "dispatch_equipment", "publish_kpis"]
    },
    {
      agent: "Safety & Compliance",
      count: 7,
      tools: ["log_incident", "start_checklist", "broadcast_alert", "lockout_tagout_request", "create_corrective_action", "retrieve_sds", "near_miss_capture"]
    }
  ];

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          Warehouse Operational Assistant
        </Typography>
        <Typography variant="h5" component="h2" gutterBottom color="text.secondary">
          Developer Guide & Implementation Documentation
        </Typography>
        <Typography variant="body1" sx={{ mb: 2 }}>
          A comprehensive guide for developers taking this NVIDIA Blueprint-aligned multi-agent warehouse assistant to the next level.
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 3 }}>
          <Chip label="Production Ready" color="success" />
          <Chip label="NVIDIA NIMs" color="primary" />
          <Chip label="MCP Framework" color="secondary" />
          <Chip label="Multi-Agent" color="info" />
          <Chip label="Hybrid RAG" color="warning" />
        </Box>
      </Box>

      {/* Quick Start Alert */}
      <Alert severity="info" sx={{ mb: 4 }}>
        <AlertTitle>🚀 Quick Start</AlertTitle>
        This system is production-ready with comprehensive documentation. Follow the quick start guide below to get up and running in minutes.
      </Alert>

      {/* Quick Start Section */}
      <Accordion expanded={expandedSection === 'overview'} onChange={handleChange('overview')}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <RocketIcon color="primary" />
            Quick Start Guide
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            {quickStartSteps.map((step) => (
              <Grid item xs={12} md={6} key={step.step}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Step {step.step}: {step.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {step.description}
                    </Typography>
                    <Paper sx={{ p: 2, bgcolor: 'grey.100', fontFamily: 'monospace', fontSize: '0.875rem' }}>
                      {step.code}
                    </Paper>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Architecture Overview */}
      <Accordion expanded={expandedSection === 'architecture'} onChange={handleChange('architecture')}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ArchitectureIcon color="primary" />
            System Architecture
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            {architectureComponents.map((component, index) => (
              <Grid item xs={12} md={6} key={index}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                      {component.icon}
                      <Typography variant="h6">{component.name}</Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {component.description}
                    </Typography>
                    <Chip 
                      label={component.status} 
                      color={component.status.includes('✅') ? 'success' : 'warning'}
                      size="small"
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* API Reference */}
      <Accordion expanded={expandedSection === 'api'} onChange={handleChange('api')}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ApiIcon color="primary" />
            API Reference
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            {apiEndpoints.map((category, index) => (
              <Grid item xs={12} md={6} key={index}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {category.category}
                    </Typography>
                    <List dense>
                      {category.endpoints.map((endpoint, epIndex) => (
                        <ListItem key={epIndex} sx={{ px: 0 }}>
                          <ListItemText
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Chip 
                                  label={endpoint.method} 
                                  size="small" 
                                  color={endpoint.method === 'GET' ? 'success' : 'primary'}
                                />
                                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                                  {endpoint.path}
                                </Typography>
                              </Box>
                            }
                            secondary={endpoint.description}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Agent Tools Overview */}
      <Accordion expanded={expandedSection === 'tools'} onChange={handleChange('tools')}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <BuildIcon color="primary" />
            Agent Tools Overview
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            {toolsOverview.map((agent, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {agent.agent}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {agent.count} Action Tools
                    </Typography>
                    <List dense>
                      {agent.tools.map((tool, toolIndex) => (
                        <ListItem key={toolIndex} sx={{ px: 0 }}>
                          <ListItemIcon>
                            <CodeIcon fontSize="small" />
                          </ListItemIcon>
                          <ListItemText 
                            primary={tool}
                            primaryTypographyProps={{ variant: 'body2', sx: { fontFamily: 'monospace' } }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Implementation Journey */}
      <Accordion expanded={expandedSection === 'journey'} onChange={handleChange('journey')}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SchoolIcon color="primary" />
            Implementation Journey & Next Steps
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ mb: 3 }}>
            <Typography variant="h5" gutterBottom>
              🎯 Current State: Production-Ready Foundation
            </Typography>
            <Typography variant="body1" sx={{ mb: 2 }}>
              This Warehouse Operational Assistant represents a <strong>production-grade implementation</strong> of NVIDIA's AI Blueprint architecture. 
              We've successfully built a multi-agent system with comprehensive tooling, advanced reasoning capabilities, and enterprise-grade security.
            </Typography>
            <Alert severity="success" sx={{ mb: 3 }}>
              <AlertTitle>✅ What's Complete</AlertTitle>
              <Typography variant="body2">
                • Multi-agent orchestration with LangGraph + MCP integration<br/>
                • NVIDIA NIMs integration (Llama 3.1 70B + NV-EmbedQA-E5-v5)<br/>
                • 23 production-ready action tools across 3 specialized agents<br/>
                • Advanced reasoning engine with 5 reasoning types<br/>
                • Hybrid RAG system with PostgreSQL/TimescaleDB + Milvus<br/>
                • Complete security stack with JWT/OAuth2 + RBAC<br/>
                • Comprehensive monitoring with Prometheus/Grafana<br/>
                • React frontend with Material-UI and real-time chat interface
              </Typography>
            </Alert>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom color="primary">
                    🚀 Immediate Development Opportunities
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemText 
                        primary="Complete MCP Adapter Migration" 
                        secondary="Migrate WMS, IoT, RFID/Barcode, and Time Attendance adapters to MCP framework for dynamic tool discovery"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="React Native Mobile App" 
                        secondary="Build field operations app for warehouse workers with offline capabilities"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Real Workforce Integration" 
                        secondary="Replace simulated workforce data with actual workforce management systems"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Production Kubernetes Deployment" 
                          secondary="Complete Helm charts and production-grade Kubernetes configurations"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Telemetry Data Integration" 
                        secondary="Fix time window issues and integrate real equipment telemetry streams"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom color="secondary">
                    🔧 Advanced Technical Enhancements
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemText 
                        primary="ML-Powered Predictive Analytics" 
                        secondary="Implement machine learning models for demand forecasting and optimization"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Real-time Event Streaming" 
                        secondary="Kafka integration for real-time data processing and event-driven architecture"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Edge Computing Deployment" 
                        secondary="Deploy lightweight agents to edge devices for local processing and reduced latency"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Multi-tenant Architecture" 
                        secondary="Support multiple warehouse organizations with data isolation and custom configurations"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Advanced Security Features" 
                        secondary="Implement zero-trust security, advanced threat detection, and compliance automation"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Box sx={{ mt: 4 }}>
            <Typography variant="h5" gutterBottom>
              🛠️ Development Roadmap
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Card variant="outlined" sx={{ bgcolor: 'success.light', color: 'success.contrastText' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Phase 1: Foundation</Typography>
                    <Typography variant="body2">✅ Complete</Typography>
                    <Typography variant="body2">Core architecture, agents, and MCP framework</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card variant="outlined" sx={{ bgcolor: 'warning.light', color: 'warning.contrastText' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Phase 2: Integration</Typography>
                    <Typography variant="body2">🔄 In Progress</Typography>
                    <Typography variant="body2">Adapter migration and mobile app development</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card variant="outlined" sx={{ bgcolor: 'info.light', color: 'info.contrastText' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Phase 3: Scale</Typography>
                    <Typography variant="body2">📋 Planned</Typography>
                    <Typography variant="body2">ML analytics, edge computing, multi-tenant</Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Technical Deep Dive */}
      <Accordion expanded={expandedSection === 'technical'} onChange={handleChange('technical')}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CodeIcon color="primary" />
            Technical Deep Dive
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ mb: 3 }}>
            <Typography variant="h5" gutterBottom>
              🏗️ Architecture Deep Dive
            </Typography>
            <Typography variant="body1" sx={{ mb: 2 }}>
              This system implements NVIDIA's AI Blueprint architecture with several key innovations and production-ready patterns.
            </Typography>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🤖 Multi-Agent Architecture
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    The system uses LangGraph for orchestration with specialized agents:
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="Planner/Router Agent" 
                        secondary="Intent classification, context management, response synthesis"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Equipment & Asset Operations Agent" 
                        secondary="8 tools for inventory, equipment, and asset management"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Operations Coordination Agent" 
                        secondary="8 tools for workforce, waves, scheduling, and KPIs"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Safety & Compliance Agent" 
                        secondary="7 tools for incidents, checklists, alerts, and compliance"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🔧 MCP Framework Innovation
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    Our MCP implementation provides dynamic tool discovery and execution:
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="Dynamic Tool Discovery" 
                        secondary="Automatic registration and discovery of available tools"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Intelligent Tool Binding" 
                        secondary="Context-aware tool selection and execution planning"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Advanced Routing" 
                        secondary="Performance, accuracy, cost, and latency-based routing"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Comprehensive Validation" 
                        secondary="Input validation, error handling, and rollback mechanisms"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              🧠 Advanced Reasoning Engine
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={3}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>Chain-of-Thought</Typography>
                    <Typography variant="body2">Step-by-step reasoning with intermediate conclusions</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>Multi-Hop Reasoning</Typography>
                    <Typography variant="body2">Complex reasoning across multiple knowledge domains</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>Scenario Analysis</Typography>
                    <Typography variant="body2">What-if analysis and scenario planning</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>Pattern Recognition</Typography>
                    <Typography variant="body2">Learning from historical patterns and trends</Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>

          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              🗄️ Data Architecture
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>PostgreSQL/TimescaleDB</Typography>
                    <Typography variant="body2">Structured data with time-series capabilities for IoT telemetry</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>Milvus Vector DB</Typography>
                    <Typography variant="body2">GPU-accelerated semantic search with NVIDIA cuVS</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>Redis Cache</Typography>
                    <Typography variant="body2">Session management and intelligent caching with LRU/LFU policies</Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Resources */}
      <Accordion expanded={expandedSection === 'resources'} onChange={handleChange('resources')}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ArticleIcon color="primary" />
            Resources & Documentation
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📚 Documentation
                  </Typography>
                  <List>
                    <ListItem>
                      <Button 
                        variant="outlined" 
                        fullWidth 
                        startIcon={<BuildIcon />}
                        onClick={() => navigate('/documentation/mcp-integration')}
                        sx={{ mb: 1 }}
                      >
                        MCP Integration Guide
                      </Button>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Complete MCP framework documentation with implementation phases, API reference, and best practices
                      </Typography>
                    </ListItem>
                    <ListItem>
                      <Button 
                        variant="outlined" 
                        fullWidth 
                        startIcon={<ApiIcon />}
                        onClick={() => navigate('/documentation/api-reference')}
                        sx={{ mb: 1 }}
                      >
                        API Reference
                      </Button>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Comprehensive API documentation with endpoints, request examples, and error handling
                      </Typography>
                    </ListItem>
                    <ListItem>
                      <Button 
                        variant="outlined" 
                        fullWidth 
                        startIcon={<RocketIcon />}
                        onClick={() => navigate('/documentation/deployment')}
                        sx={{ mb: 1 }}
                      >
                        Deployment Guide
                      </Button>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Production deployment instructions for Docker, Kubernetes, and Helm
                      </Typography>
                    </ListItem>
                    <ListItem>
                      <Button 
                        variant="outlined" 
                        fullWidth 
                        startIcon={<ArchitectureIcon />}
                        onClick={() => navigate('/documentation/architecture')}
                        sx={{ mb: 1 }}
                      >
                        Architecture Diagrams
                      </Button>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        System architecture and flow diagrams with component descriptions
                      </Typography>
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🔗 External Resources
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemText 
                        primary="NVIDIA NIMs Documentation" 
                        secondary="Official NVIDIA NIMs API documentation"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="MCP Specification" 
                        secondary="Model Context Protocol specification"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="LangGraph Documentation" 
                        secondary="Multi-agent orchestration framework"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="FastAPI Documentation" 
                        secondary="Modern Python web framework"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Footer */}
      <Box sx={{ mt: 4, pt: 3, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="body2" color="text.secondary" align="center">
          Warehouse Operational Assistant - Built with NVIDIA NIMs, MCP Framework, and Modern Web Technologies
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 2 }}>
          <Button 
            variant="outlined" 
            startIcon={<GitHubIcon />}
            href="https://github.com/T-DevH/warehouse-operational-assistant"
            target="_blank"
            rel="noopener noreferrer"
          >
            View Source
          </Button>
          <Button variant="outlined" startIcon={<DashboardIcon />}>
            Live Demo
          </Button>
        </Box>
      </Box>
    </Box>
  );
};

export default Documentation;
