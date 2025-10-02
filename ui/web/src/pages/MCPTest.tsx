import React from 'react';
import { Box, Typography, Container } from '@mui/material';
import MCPTestingPanel from '../components/MCPTestingPanel';

const MCPTestPage: React.FC = () => {
  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          MCP Testing Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Test and explore the Model Context Protocol (MCP) framework integration
        </Typography>
      </Box>
      
      <MCPTestingPanel />
    </Container>
  );
};

export default MCPTestPage;
