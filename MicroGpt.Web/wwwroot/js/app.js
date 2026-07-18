// Minimal interop: trigger a client-side file download from bytes produced in .NET.
// No network involved — the blob is created and revoked entirely in this tab.
window.appInterop = {
    downloadFile: function (fileName, base64, mime) {
        const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
        const blob = new Blob([bytes], { type: mime || "application/octet-stream" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    }
};
