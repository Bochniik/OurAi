# Self-hosted GPU runner (Windows)

This setup lets trusted GitHub Actions jobs run on the local Windows machine and use its NVIDIA GPU for OurAI experiments.

## Security first

`Bochniik/OurAi` is currently public. A self-hosted runner can execute commands on the machine that hosts it, so **never attach this runner to workflows triggered by untrusted pull requests or forks**.

The included GPU smoke workflow is intentionally `workflow_dispatch` only and targets a dedicated `ourai-gpu` label. Keep it that way unless the repository is made private or a stronger isolation model is introduced.

For initial testing, prefer running the runner interactively with `run.cmd` under a normal Windows user instead of giving it administrator privileges.

## 1. Create the runner in GitHub

Open the repository and go to:

`Settings -> Actions -> Runners -> New self-hosted runner`

Choose:

- Windows
- x64

GitHub will show current download and configuration commands. Use those exact commands because the runner version and registration token change over time. The registration token is time-limited.

GitHub recommends `C:\actions-runner` on Windows.

## 2. Add the dedicated label

When configuring the runner, make sure it receives the custom label:

`ourai-gpu`

If using the command shown by GitHub manually, the configuration command can include:

```powershell
.\config.cmd --url https://github.com/Bochniik/OurAi --token <TOKEN_FROM_GITHUB> --labels ourai-gpu
```

Do not commit or share the registration token.

Suggested runner name:

`ourai-gtx1650`

## 3. Start the runner

For the first test, start it interactively:

```powershell
cd C:\actions-runner
.\run.cmd
```

When connected successfully, the terminal should report that it is listening for jobs.

The PC must be powered on and the runner process must be active for GitHub to assign work. If the machine is offline, jobs wait for a matching runner to come online and eventually time out if none does.

## 4. Point the workflow at the existing CUDA environment

The workflow first looks for an `OURAI_PYTHON` environment variable, then falls back to the validated desktop virtual environment paths.

Recommended user environment variable:

```powershell
[Environment]::SetEnvironmentVariable(
    "OURAI_PYTHON",
    "$env:USERPROFILE\Desktop\OurAI\.venv\Scripts\python.exe",
    "User"
)
```

Restart the runner process after changing environment variables so it inherits the new value.

## 5. Run the smoke test

After this branch is merged, open:

`Actions -> GPU runner smoke test -> Run workflow`

The job should:

1. check out the repository,
2. run `nvidia-smi`,
3. locate the OurAI Python environment,
4. import PyTorch,
5. verify `torch.cuda.is_available()`,
6. run a small FP16 matrix multiplication on the GPU.

A successful run ends with:

`SUCCESS: self-hosted runner can execute PyTorch CUDA workloads.`

## 6. Service mode later

Once the interactive setup is proven stable, the Windows runner can be reconfigured to run as a service so it starts automatically when Windows boots. On Windows this choice is made during runner configuration and requires an elevated shell.

For a personal development machine, do not rush this step. Interactive mode makes it obvious when GitHub Actions has access to the machine and is safer while the repository remains public.

## Next step after smoke testing

Only after the smoke workflow is proven should we add a separate trusted training workflow. That workflow should remain restricted to trusted triggers and the `ourai-gpu` label, cap runtime, record experiment metadata, and upload only small metrics/artifacts rather than committing checkpoints directly to Git.
