document.addEventListener('DOMContentLoaded', () => {
    const sourceLangValue = document.getElementById('source-lang-value');
    const targetLangValue = document.getElementById('target-lang-value');
    const sourceLangName = document.getElementById('source-lang-name');
    const targetLangName = document.getElementById('target-lang-name');
    const swapBtn = document.getElementById('swap-btn');
    const sourceInput = document.getElementById('source-input');
    const translationOutput = document.getElementById('translation-output');
    const sourceCharCount = document.getElementById('source-char-count');
    const clearInputBtn = document.getElementById('clear-input-btn');
    const copyOutputBtn = document.getElementById('copy-output-btn');
    const translationForm = document.getElementById('translation-form');

    function swapLanguages() {
        const tempLang = sourceLangValue.value;
        sourceLangValue.value = targetLangValue.value;
        targetLangValue.value = tempLang;

        sourceLangName.textContent = sourceLangValue.value;
        targetLangName.textContent = targetLangValue.value;

        const tempText = sourceInput.value;
        sourceInput.value = translationOutput.textContent.trim() !== 'Translation appears here' ? translationOutput.textContent.trim() : '';
        
        if (tempText.trim() !== '') {
            translationOutput.innerHTML = `<span class="placeholder-text">${tempText}</span>`;
        } else {
            translationOutput.innerHTML = '<span class="placeholder-text">Translation appears here</span>';
        }
        
        updateCharCount();
    }

    function updateCharCount() {
        sourceCharCount.textContent = sourceInput.value.length;
    }

    function clearInput() {
        sourceInput.value = '';
        updateCharCount();
    }

    function copyOutput() {
        const textToCopy = translationOutput.textContent.trim();
        if (textToCopy && textToCopy !== 'Translation appears here') {
            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalIcon = copyOutputBtn.innerHTML;
                copyOutputBtn.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    copyOutputBtn.innerHTML = originalIcon;
                }, 1500);
            });
        }
    }

    swapBtn.addEventListener('click', swapLanguages);
    sourceInput.addEventListener('input', updateCharCount);
    clearInputBtn.addEventListener('click', clearInput);
    copyOutputBtn.addEventListener('click', copyOutput);

    // Initialize character count on page load
    updateCharCount();

    translationForm.addEventListener('submit', (e) => {
        if (sourceInput.value.trim() === '') {
            e.preventDefault();
        }
    });
});


