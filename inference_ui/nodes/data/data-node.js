// ============================================
// Data Node Type (Data category)
// ============================================

(function() {

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

NodeRegistry.register('data', {
    label: 'Data',
    category: 'Data',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'data', label: 'File' }
    ],

    defaultConfig: {
        title: 'Data',
        menuTag: { en: 'Data', ko: '데이터', ja: 'データ', zh: '数据', fr: 'Données', de: 'Daten', es: 'Datos', it: 'Dati', pt: 'Dados', nl: 'Gegevens', ru: 'Данные', ar: 'بيانات', hi: 'डेटा', tr: 'Veri', pl: 'Dane', cs: 'Data', sv: 'Data', da: 'Data', no: 'Data', fi: 'Data', el: 'Δεδομένα', hu: 'Adatok', ro: 'Date', uk: 'Дані', vi: 'Dữ liệu', th: 'ข้อมูล', id: 'Data' },
        description: { en: 'Load file or dataset from local storage or URL', ko: '로컬 저장소나 URL에서 파일 또는 데이터셋 로드', ja: 'ローカルストレージまたはURLからファイルやデータセットを読み込む', zh: '从本地存储或URL加载文件或数据集', fr: 'Charger un fichier ou un jeu de données depuis le stockage local ou une URL', de: 'Datei oder Datensatz aus lokalem Speicher oder URL laden', es: 'Cargar archivo o conjunto de datos desde almacenamiento local o URL', it: 'Carica file o set di dati da archivio locale o URL', pt: 'Carregar arquivo ou conjunto de dados do armazenamento local ou URL', nl: 'Bestand of dataset laden vanuit lokale opslag of URL', ru: 'Загрузка файла или набора данных из локального хранилища или URL', ar: 'تحميل ملف أو مجموعة بيانات من التخزين المحلي أو عنوان URL', hi: 'स्थानीय संग्रहण या URL से फ़ाइल या डेटासेट लोड करें', tr: 'Yerel depolamadan veya URL üzerinden dosya veya veri seti yükle', pl: 'Załaduj plik lub zbiór danych z lokalnego magazynu lub adresu URL', cs: 'Načíst soubor nebo datovou sadu z místního úložiště nebo URL', sv: 'Ladda fil eller dataset från lokal lagring eller URL', da: 'Indlæs fil eller datasæt fra lokalt lager eller URL', no: 'Last inn fil eller datasett fra lokal lagring eller URL', fi: 'Lataa tiedosto tai tietojoukko paikallisesta tallennustilasta tai URL-osoitteesta', el: 'Φόρτωση αρχείου ή συνόλου δεδομένων από τοπική αποθήκευση ή URL', hu: 'Fájl vagy adatkészlet betöltése helyi tárolóból vagy URL-ből', ro: 'Încarcă fișier sau set de date din stocarea locală sau URL', uk: 'Завантаження файлу або набору даних з локального сховища або URL', vi: 'Tải tệp hoặc tập dữ liệu từ bộ nhớ cục bộ hoặc URL', th: 'โหลดไฟล์หรือชุดข้อมูลจากที่เก็บข้อมูลในเครื่องหรือ URL', id: 'Muat file atau dataset dari penyimpanan lokal atau URL' },
        status: 'completed',
        portValues: { out: null }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const fileData = node.portValues && node.portValues.out;

        let bodyContent;
        if (fileData && fileData.fileName) {
            const preview = fileData.textContent
                ? helpers.escapeHtml(fileData.textContent.substring(0, 200)) + (fileData.textContent.length > 200 ? '...' : '')
                : '';
            bodyContent = `
                <div class="ng-data-file-info">
                    <span class="ng-data-file-icon"></span>
                    <span class="ng-data-file-name">${helpers.escapeHtml(fileData.fileName)}</span>
                    <span class="ng-data-file-size">${formatFileSize(fileData.fileSize || 0)}</span>
                    <button class="ng-data-remove-btn ng-interactive" title="Remove">\u00d7</button>
                </div>
                ${preview ? `<div class="ng-data-preview">${preview}</div>` : ''}
            `;
        } else {
            bodyContent = `
                <button class="ng-data-browse-btn ng-interactive">Browse File</button>
            `;
        }

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-data-body">
                ${bodyContent}
                <input type="file" class="ng-data-file-input" accept=".csv,.xml,.json,.txt,.pdf,.doc,.docx,.xlsx,.xls" hidden>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="data" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    afterRender(el, node, helpers) {
        const fileInput = el.querySelector('.ng-data-file-input');
        const browseBtn = el.querySelector('.ng-data-browse-btn');
        const removeBtn = el.querySelector('.ng-data-remove-btn');

        if (browseBtn) {
            browseBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                fileInput.click();
            });
        }

        if (removeBtn) {
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                node.portValues.out = null;
                helpers.rerender(node);
            });
        }

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = async (ev) => {
                try {
                    const resp = await fetch('/api/data/upload', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name: file.name, data: ev.target.result })
                    });
                    if (!resp.ok) throw new Error('Upload failed');
                    const result = await resp.json();

                    node.portValues.out = {
                        fileName: result.fileName,
                        fileSize: result.fileSize,
                        uploadId: result.uploadId,
                        textContent: result.textContent,
                        extractionMethod: result.extractionMethod
                    };
                    helpers.rerender(node);
                } catch (err) {
                    console.error('Data node upload failed:', err);
                    alert('Failed to upload file');
                }
            };
            reader.readAsDataURL(file);
        });
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});

})();
