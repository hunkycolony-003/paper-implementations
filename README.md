# Paper Implementations

A collection repository for paper implementation projects. Each paper lives in its own directory with its own code, dependency file, and setup notes.

## Structure

```text
paper-implementations/
├── README.md
├── .gitignore
└── Transformer/
    ├── README.md
    ├── requirements.txt
    └── model/
```

## Repository strategy

This repository is organized as a monorepo for paper implementations. That works well when:

- each paper is a self-contained folder
- you want one place to browse all implementations
- you prefer a single GitHub profile surface for related work

If a future implementation becomes large, needs independent releases, or requires a very different toolchain, it can be split into its own repository later.

## General workflow

For each paper implementation:

1. Create a dedicated subdirectory.
2. Keep source code, notes, and a local dependency file inside that directory.
3. Do not commit local virtual environments or machine-specific files.
4. Add a short README describing the paper, setup, and current status.
