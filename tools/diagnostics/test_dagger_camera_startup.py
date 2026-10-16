from pathlib import Path
import os
import subprocess

import pytest


PROJECT = Path(__file__).resolve().parents[2]
POLICY_SERIAL = "151222073444"


@pytest.fixture
def launcher(tmp_path):
    scripts = tmp_path / "script"
    scripts.mkdir()
    install = tmp_path / "install"
    install.mkdir()
    (install / "setup.bash").write_text("")
    sdk_setup = tmp_path / "sdk.bash"
    sdk_setup.write_text("")
    script = scripts / "start_dagger_hw_test.sh"
    script.write_text((PROJECT / "script" / script.name).read_text().replace(
        "/opt/bxi/bxi_ros2_pkg/setup.bash", str(sdk_setup)))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    ros2 = binaries / "ros2"
    ros2.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CAMERA_TEST_CALLS"\n'
        'if [[ "$1" == run ]]; then exit "$CAMERA_TEST_PREFLIGHT_EXIT"; fi\n'
        'exit 0\n'
    )
    ros2.chmod(0o755)
    pgrep = binaries / "pgrep"
    pgrep.write_text("#!/bin/bash\nexit 0\n")
    pgrep.chmod(0o755)
    calls = tmp_path / "calls.txt"
    environment = dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}",
                       CAMERA_TEST_CALLS=str(calls), CAMERA_TEST_PREFLIGHT_EXIT="0")
    return script, environment, calls


@pytest.mark.parametrize("arguments, serial", [([], POLICY_SERIAL), (["another"], "another")])
def test_selected_camera_starts_despite_unrelated_camera_process(launcher, arguments, serial):
    script, environment, calls = launcher
    result = subprocess.run(["bash", str(script), *arguments], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = calls.read_text().splitlines()
    assert commands[0] == f"run bxi_depth_camera cameras-inspect --serial {serial} --require-depth"
    assert len(commands) == 2
    assert commands[1].startswith("launch bxi_example_py_elf3 example_demo_hw.launch.py ")
    assert f"depth_camera_serial_no:={serial}" in commands[1]


def test_failed_selected_camera_prevents_hardware_start(launcher):
    script, environment, calls = launcher
    environment["CAMERA_TEST_PREFLIGHT_EXIT"] = "1"
    result = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True)
    assert result.returncode == 1
    assert len(calls.read_text().splitlines()) == 1
    assert "hardware was not started" in result.stderr


def test_extra_arguments_do_not_start_camera_or_hardware(launcher):
    script, environment, calls = launcher
    result = subprocess.run(["bash", str(script), "first", "second"], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert not calls.exists()


def test_source_and_installed_launch_defaults_match_selected_camera():
    for path in (PROJECT / "src/bxi_example_py_elf3/launch/example_demo_hw.launch.py",
                 PROJECT / "install/share/bxi_example_py_elf3/launch/example_demo_hw.launch.py"):
        assert f'"depth_camera_serial_no",\n                default_value="{POLICY_SERIAL}"' in path.read_text()
