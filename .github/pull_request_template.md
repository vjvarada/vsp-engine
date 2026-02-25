## Summary
<!-- What does this PR do? -->

## Type of change
- [ ] Bug fix
- [ ] New feature / workflow step
- [ ] Refactor
- [ ] Docs / config only

## Testing
- [ ] Frontend: `pnpm exec tsc --noEmit` passes
- [ ] Backend: `pytest tests/` passes
- [ ] Mesh watertight check tested (if mesh pipeline changed)
- [ ] No real DICOM patient data in diff

## Safety checklist (if medical pipeline touched)
- [ ] DICOM anonymization still applied at upload boundary
- [ ] "FOR PLANNING PURPOSES ONLY" watermark preserved on exports
- [ ] `topology.isClosed()` checked before STL is offered for download
- [ ] No PHI logged
