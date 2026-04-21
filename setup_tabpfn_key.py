"""
One-time setup: save your TabPFN API key so it never opens a browser again.

Usage
-----
    python setup_tabpfn_key.py YOUR_API_KEY_HERE

Get your API key at: https://ux.priorlabs.ai/account/licenses
After running this once, TabPFN will authenticate silently on every future run.
"""
import sys
import os
import json
import asyncio

# Fix Windows asyncio event loop before anything touches sockets.
# Python 3.8+ defaults to ProactorEventLoop which breaks TabPFN's
# localhost OAuth callback server (WinError 10038).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_tabpfn_key.py YOUR_API_KEY_HERE")
        print("Get your key at: https://ux.priorlabs.ai/account/licenses")
        sys.exit(1)

    api_key = sys.argv[1].strip()
    print(f"[setup_tabpfn] Saving API key ({api_key[:6]}...)")

    saved = False

    # ── Method 1: tabpfn's own UserDataClient ────────────────────────────────
    if not saved:
        try:
            from tabpfn.utils.user_data_client import UserDataClient
            client = UserDataClient()
            client.save_api_key(api_key)
            saved = True
            print("[OK] Key saved via tabpfn.utils.user_data_client.UserDataClient")
        except Exception as e:
            print(f"  Method 1 (UserDataClient) failed: {e}")

    # ── Method 2: write directly to platformdirs user-data path ─────────────
    if not saved:
        try:
            import platformdirs
            config_dir = platformdirs.user_data_dir("tabpfn", "priorlabs")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "config.json")
            config = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
            config["api_key"] = api_key
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            saved = True
            print(f"[OK] Key written to {config_path}")
        except Exception as e:
            print(f"  Method 2 (platformdirs config file) failed: {e}")

    # ── Method 3: write to common fallback paths ─────────────────────────────
    if not saved:
        candidates = [
            os.path.join(os.path.expanduser("~"), ".tabpfn", "config.json"),
            os.path.join(os.environ.get("APPDATA", ""), "tabpfn", "config.json"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "tabpfn", "config.json"),
        ]
        for path in candidates:
            if not path or path.startswith(os.sep + "tabpfn"):
                continue
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                config = {}
                if os.path.exists(path):
                    with open(path) as f:
                        config = json.load(f)
                config["api_key"] = api_key
                with open(path, "w") as f:
                    json.dump(config, f, indent=2)
                saved = True
                print(f"[OK] Key written to {path}")
                break
            except Exception as e:
                print(f"  Method 3 path {path} failed: {e}")

    # ── Method 4: permanent Windows environment variable (last resort) ────────
    if not saved:
        ret = os.system(f'setx TABPFN_API_KEY "{api_key}"')
        if ret == 0:
            saved = True
            print("[OK] Key saved as permanent Windows env var TABPFN_API_KEY.")
            print("     You must RESTART your terminal for it to take effect.")
        else:
            print("  Method 4 (setx) failed.")

    if not saved:
        print("\n[FAILED] Could not save API key automatically.")
        print("Manually add this to your system environment variables:")
        print(f"  Variable name : TABPFN_API_KEY")
        print(f"  Value         : {api_key}")
        sys.exit(1)

    # ── Verify: import + quick smoke test ────────────────────────────────────
    print("\n[verify] Testing TabPFN import ...")
    os.environ["TABPFN_API_KEY"] = api_key   # set for this process too
    try:
        from tabpfn import TabPFNRegressor
        print("[OK] TabPFN imported successfully — no browser should open now.")
        print("\nYou can now run:")
        print("    python experiments/f1/run_baselines.py")
    except Exception as e:
        print(f"[WARN] Import check failed: {e}")


if __name__ == "__main__":
    main()
