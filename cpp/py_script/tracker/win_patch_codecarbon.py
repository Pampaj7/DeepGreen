import sys

# Applica patch solo su Windows
if sys.platform.startswith("win"):
    import codecarbon.core.cpu as cc_cpu
    import codecarbon.core.util as cc_util

    try:
        import winreg
    except ImportError:
        winreg = None

    def _safe_detect_cpu_model():
        """
        Rileva il modello CPU da Windows Registry.
        Evita chiamate lente/bloccanti a PowerShell/WMIC.
        """
        if winreg is None:
            return "Unknown CPU (winreg not available)"
        try:
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return cpu_name.strip()
        except Exception as e:
            return f"Unknown CPU (registry read failed: {e})"

    # Applica la patch
    cc_util.detect_cpu_model = _safe_detect_cpu_model
    cc_cpu.detect_cpu_model = _safe_detect_cpu_model

    print("[Patch] CodeCarbon CPU detection patched for Windows.")
else:
    print("[Patch] Not on Windows — no changes applied.")
