// ============================================
// Image Node Type (Data category)
// ============================================

(function() {

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

NodeRegistry.register('image', {
    label: 'Image',
    category: 'Data',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'image', label: 'Image' }
    ],

    defaultConfig: {
        title: 'Image',
        menuTag: { en: 'Image', ko: '이미지', ja: '画像', zh: '图像', fr: 'Image', de: 'Bild', es: 'Imagen', it: 'Immagine', pt: 'Imagem', nl: 'Afbeelding', ru: 'Изображение', ar: 'صورة', hi: 'छवि', tr: 'Görsel', pl: 'Obraz', cs: 'Obrázek', sv: 'Bild', da: 'Billede', no: 'Bilde', fi: 'Kuva', el: 'Εικόνα', hu: 'Kép', ro: 'Imagine', uk: 'Зображення', vi: 'Hình ảnh', th: 'รูปภาพ', id: 'Gambar' },
        description: { en: 'Load and output image data for processing', ko: '이미지 데이터를 로드하고 출력', ja: '処理用の画像データを読み込んで出力する', zh: '加载并输出用于处理的图像数据', fr: 'Charger et produire des données image pour le traitement', de: 'Bilddaten zur Verarbeitung laden und ausgeben', es: 'Cargar y producir datos de imagen para procesamiento', it: 'Carica e produce dati immagine per elaborazione', pt: 'Carregar e produzir dados de imagem para processamento', nl: 'Beeldgegevens laden en uitvoeren voor verwerking', ru: 'Загрузка и вывод данных изображений для обработки', ar: 'تحميل وإخراج بيانات الصورة للمعالجة', hi: 'प्रसंस्करण के लिए छवि डेटा लोड और आउटपुट करें', tr: 'İşleme için görsel verilerini yükle ve çıktıla', pl: 'Załaduj i wyślij dane obrazu do przetwarzania', cs: 'Načíst a vydat obrazová data ke zpracování', sv: 'Ladda och mata ut bilddata för bearbetning', da: 'Indlæs og udlæs billeddata til behandling', no: 'Last inn og utdata bildedata for behandling', fi: 'Lataa ja tulosta kuvatiedot käsittelyä varten', el: 'Φόρτωση και εξαγωγή δεδομένων εικόνας για επεξεργασία', hu: 'Képadatok betöltése és kimenete feldolgozáshoz', ro: 'Încarcă și produce date de imagine pentru procesare', uk: 'Завантаження та виведення даних зображень для обробки', vi: 'Tải và xuất dữ liệu hình ảnh để xử lý', th: 'โหลดและส่งออกข้อมูลรูปภาพสำหรับการประมวลผล', id: 'Muat dan keluarkan data gambar untuk pemrosesan' },
        status: 'completed',
        portValues: { out: null }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const fileData = node.portValues && node.portValues.out;

        let bodyContent;
        if (fileData && fileData.uploadId) {
            bodyContent = `
                <div class="ng-image-preview">
                    <img src="/uploads/${helpers.escapeHtml(fileData.uploadId)}"
                         alt="${helpers.escapeHtml(fileData.fileName || 'image')}">
                </div>
                <div class="ng-data-file-info">
                    <span class="ng-data-file-icon">🖼</span>
                    <span class="ng-data-file-name">${helpers.escapeHtml(fileData.fileName)}</span>
                    <span class="ng-data-file-size">${formatFileSize(fileData.fileSize || 0)}</span>
                    <button class="ng-data-remove-btn ng-interactive" title="Remove">\u00d7</button>
                </div>
            `;
        } else {
            bodyContent = `
                <button class="ng-data-browse-btn ng-interactive">Browse Image</button>
            `;
        }

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-data-body">
                ${bodyContent}
                <input type="file" class="ng-data-file-input" accept=".png,.jpg,.jpeg,.gif,.webp,.bmp,.svg" hidden>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="image" data-node-id="${node.id}"></div>
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
                        isImage: true
                    };
                    helpers.rerender(node);
                } catch (err) {
                    console.error('Image node upload failed:', err);
                    alert('Failed to upload image');
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
