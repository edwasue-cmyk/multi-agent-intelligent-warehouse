# Dependabot Configuration Guide

## Overview
This repository uses GitHub Dependabot to automatically manage dependency updates across multiple package ecosystems.

## Configuration Features

### 🔄 Auto-Merge Strategy
- **Security Updates**: Automatically merged for critical security patches
- **Patch Updates**: Automatically merged for bug fixes (patch versions)
- **Minor Updates**: Manual review required
- **Major Updates**: Manual review required with special attention to breaking changes

### 📅 Update Schedule
- **Python/NPM/Docker/GitHub Actions**: Weekly updates (Mondays at 9:00 AM)
- **Helm Charts**: Monthly updates (first Monday at 9:00 AM)

### 🏷️ Labeling System
All Dependabot PRs are automatically labeled with:
- `dependencies` - General dependency updates
- `security` - Security-related updates
- `python`/`javascript`/`docker`/`github-actions`/`helm` - Ecosystem-specific labels
- `backend`/`frontend`/`infrastructure`/`ci-cd`/`kubernetes` - Component-specific labels

### 🚫 Ignored Updates
Major version updates are ignored for critical packages to prevent breaking changes:

#### Python Packages
- `fastapi` - Core API framework
- `uvicorn` - ASGI server
- `langchain` - AI/ML framework

#### JavaScript Packages
- `react` - Frontend framework
- `react-dom` - React DOM bindings
- `@mui/material` - Material-UI core
- `@mui/icons-material` - Material-UI icons
- `typescript` - TypeScript compiler

## How to Handle Dependabot PRs

### ✅ Auto-Merged Updates
These updates are automatically merged:
- Security patches
- Bug fixes (patch versions)
- Docker image patches
- GitHub Actions patches

### 🔍 Manual Review Required
These updates require manual review:
- Minor version updates
- Major version updates
- Helm chart updates

### 📋 Review Checklist
When reviewing Dependabot PRs:

1. **Check Release Notes**
   - Review changelog for breaking changes
   - Look for new features or improvements
   - Check for deprecated functionality

2. **Test the Update**
   - Run existing tests
   - Test critical functionality
   - Check for performance impacts

3. **Update Configuration**
   - Update any configuration files if needed
   - Update documentation if APIs changed
   - Update environment variables if required

4. **Monitor After Merge**
   - Watch for any runtime issues
   - Monitor performance metrics
   - Check error logs

## Emergency Procedures

### 🚨 Security Updates
Security updates are automatically merged but should be monitored:
1. Check security advisory details
2. Verify the fix addresses the vulnerability
3. Test the application thoroughly
4. Deploy to production quickly

### 🔄 Rollback Strategy
If an update causes issues:
1. Revert the specific commit
2. Create a new PR with the previous version
3. Investigate the issue
4. Create a proper fix or find an alternative

## Configuration Files

### Main Configuration
- `.github/dependabot.yml` - Main Dependabot configuration

### Package Files Monitored
- `requirements.txt` - Python dependencies
- `ui/web/package.json` - Node.js dependencies
- `Dockerfile` - Docker base images
- `.github/workflows/*.yml` - GitHub Actions
- `helm/*/Chart.yaml` - Helm chart dependencies

## Best Practices

### 🎯 Dependency Management
- Keep dependencies up to date regularly
- Use semantic versioning constraints
- Pin critical dependencies to specific versions
- Use dependency groups for different environments

### 🔒 Security
- Enable Dependabot security alerts
- Review security updates immediately
- Use automated security scanning
- Keep security dependencies current

### 📊 Monitoring
- Monitor dependency update frequency
- Track update success rates
- Watch for breaking changes
- Maintain update documentation

## Troubleshooting

### Common Issues
1. **Build Failures**: Check for breaking changes in dependencies
2. **Test Failures**: Update tests for new API changes
3. **Performance Issues**: Monitor metrics after updates
4. **Compatibility Issues**: Check dependency compatibility matrix

### Getting Help
- Check Dependabot documentation
- Review package release notes
- Test in development environment first
- Create issues for complex updates

## Contact
For questions about dependency management:
- Create an issue with the `dependencies` label
- Tag @T-DevH for urgent security updates
- Use the dependency update template for specific requests
