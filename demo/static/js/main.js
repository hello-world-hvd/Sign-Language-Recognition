document.addEventListener('DOMContentLoaded', function() {
    const testMode = document.getElementById('testMode');
    const textMode = document.getElementById('textMode');
    const testSection = document.getElementById('testModeSection');
    const textSection = document.getElementById('textModeSection');
    const clearBtn = document.getElementById('clearText');
    const copyBtn = document.getElementById('copyText');
    const outputText = document.getElementById('outputText');

    async function startCamera() {
        await fetch('/start_camera');
    }

    async function stopCamera() {
        await fetch('/stop_camera');
    }

    testMode.addEventListener('click', async function() {
        await startCamera();
        testSection.style.display = 'block';
        textSection.style.display = 'none';
    });

    textMode.addEventListener('click', async function() {
        await startCamera();
        textSection.style.display = 'block';
        testSection.style.display = 'none';
    });

    // Thêm nút quay lại
    const backButtons = document.querySelectorAll('.back-btn');
    backButtons.forEach(btn => {
        btn.addEventListener('click', async function() {
            await stopCamera();
            testSection.style.display = 'none';
            textSection.style.display = 'none';
        });
    });

    clearBtn.addEventListener('click', function() {
        outputText.value = '';
    });

    copyBtn.addEventListener('click', function() {
        outputText.select();
        document.execCommand('copy');
        alert('Đã sao chép văn bản!');
    });
});