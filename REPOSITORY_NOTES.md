# Notes for replacing the current repository

The cleaned pipeline intentionally removes code that does not generate the reported benchmark:

- Roboflow API-key cells and credentials.
- Absolute Windows paths.
- Historical pseudo-labeling notebooks.
- Destructive leakage-cleanup cells with hard-coded filenames.
- Duplicate model definitions.
- The earlier K-fold script that tested each fold on the locked test set.
- The K-fold dummy replacement for the custom Attention U-Net.
- Test-set threshold optimization as a source of final metrics.
- Notebook-only `display()` calls.
- Unused plotting dependencies.
- Turkish-language diagnostic comments in public source files.

The final benchmark uses the validation-selected threshold and reports the locked-test result only once per dataset/model pair.

Important: `02_build_final_datasets.py` uses the same 15-image test set from the 70-image source collection and hash-matched duplicate screening. It cannot recover source-image lineage that is absent from the provided filenames or metadata. For a stronger future experiment, store a `source_id` for every original and augmented image and partition by `source_id`.
