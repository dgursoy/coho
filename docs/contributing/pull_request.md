# Pull Request Guidelines

Pull requests are the primary way to contribute code changes to Coho. Here are our guidelines for submitting PRs.

## Creating a Pull Request

Try to follow the PR title and description format below.

### **PR Title Format**

```
[TYPE] Brief description
```
Where `TYPE` is one of: `FEATURE`, `FIX`, `DOCS`, `STYLE`, `REFACTOR`, `TEST`

### **PR Description Template**

```markdown
## Description
Brief description of changes

## Related Issue
Fixes #(issue)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code style update

## Testing
Describe testing done

## Screenshots
If applicable

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests passing
```

Submit your PR and wait for it to be reviewed.

> **Note:** 
> To submit a PR:
>
> 1. Fork and branch:
>    ```bash
>    git checkout -b feature/your-feature-name  # Use descriptive branch names
>    ```
> 2. Make changes and commit:
>    ```bash
>    git commit -m "[TYPE] Brief description"  # Be clear and concise
>    ```
> 3. Push and set upstream:
>    ```bash
>    git push -u origin feature/your-feature-name  # -u links local and remote branches
>    ```
> 4. Open PR on [Coho repository](https://github.com/dgursoy/coho)
> 5. Add title, description, and related issues

## Review Process

We (plan to) use [GitHub Actions](https://github.com/features/actions) to run tests and linting. We also have a [code of conduct](code_of_conduct.md) that contributors must adhere to.
