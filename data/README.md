# Organizer-provided student resource archive

`student_resource.zip` is the original challenge resource shared with the team. It is stored as a Git LFS object, not as a normal Git blob. SHA-256: `2aefd2f8eb6f132b8933fccc1cbb98fa6356756a7581fed5914a5f26704a8bc5`. Size: 1,094,823,222 bytes.

The archive contains seven UTF-8 TSV files under `student_resource/dataset/`:

| Split | File |
| --- | --- |
| Train | `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv` |
| Test | `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` |

It also includes `student_resource/utils/validate_submission.py`, `Documentation_template.md`, and the organizer README. The setup helper extracts those files and skips `__MACOSX/` and `.DS_Store` entries.

From the repository root, run `git lfs pull` and `python scripts/prepare_dataset.py`. The extracted `student_resource/` directory is ignored. Do not commit another copy of the dataset or generated derivatives: Git LFS stores every new version as another full object and charges the repository owner for storage and downloads. Do not use external services to look up or augment the business records.
