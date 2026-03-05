// ============================================
// Composite Node Type (General category)
// Combines image + text prompt for multimodal inference
// ============================================

NodeRegistry.register('composite', {
    label: 'Composite',
    category: 'General',
    allowRef: true,

    ports: [
        { name: 'image', dir: 'in', type: 'image', label: 'Image' },
        { name: 'prompt', dir: 'in', type: 'string', label: 'Prompt' },
        { name: 'out', dir: 'out', type: 'any', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Composite',
        tool: 'view_image',
        menuTag: { en: 'Vision', ko: '비전', ja: 'ビジョン', zh: '视觉', fr: 'Vision', de: 'Vision', es: 'Visión', it: 'Visione', pt: 'Visão', nl: 'Visie', ru: 'Зрение', ar: 'رؤية', hi: 'दृष्टि', tr: 'Görüş', pl: 'Wizja', cs: 'Vize', sv: 'Vision', da: 'Vision', no: 'Visjon', fi: 'Näkö', el: 'Όραση', hu: 'Látás', ro: 'Viziune', uk: 'Зір', vi: 'Thị giác', th: 'วิชัน', id: 'Visi' },
        description: { en: 'Analyze image with text prompt via vision encoder', ko: '비전 인코더를 통해 이미지와 텍스트 프롬프트 분석', ja: 'ビジョンエンコーダーを通じてテキストプロンプトで画像を分析する', zh: '通过视觉编码器使用文本提示分析图像', fr: 'Analyser une image avec un prompt textuel via un encodeur de vision', de: 'Bild mit Textprompt über Vision-Encoder analysieren', es: 'Analizar imagen con prompt de texto mediante codificador de visión', it: 'Analizza immagine con prompt di testo tramite encoder visivo', pt: 'Analisar imagem com prompt de texto via codificador de visão', nl: 'Afbeelding analyseren met tekstprompt via vision-encoder', ru: 'Анализ изображения с текстовым запросом через визуальный кодировщик', ar: 'تحليل الصورة باستخدام نص توجيهي عبر مشفر الرؤية', hi: 'विज़न एन्कोडर के माध्यम से टेक्स्ट प्रॉम्प्ट के साथ छवि का विश्लेषण करें', tr: 'Görüntüyü vizyon kodlayıcı aracılığıyla metin istemiyle analiz et', pl: 'Analiza obrazu z promptem tekstowym przez koder wizyjny', cs: 'Analyzovat obraz s textovým promptem přes vizuální kodér', sv: 'Analysera bild med textprompt via vision-kodare', da: 'Analyser billede med tekstprompt via vision-encoder', no: 'Analyser bilde med tekstprompt via vision-enkoder', fi: 'Analysoi kuva tekstikehotteella näkökooderilla', el: 'Ανάλυση εικόνας με κειμενική εντολή μέσω κωδικοποιητή όρασης', hu: 'Kép elemzése szöveges prompttal vizuális kódolón keresztül', ro: 'Analizează imaginea cu prompt text prin codificator vizual', uk: 'Аналіз зображення з текстовим запитом через візуальний кодувальник', vi: 'Phân tích hình ảnh bằng prompt văn bản qua bộ mã hóa thị giác', th: 'วิเคราะห์รูปภาพด้วยข้อความแจ้งผ่านตัวเข้ารหัสวิชัน', id: 'Analisis gambar dengan prompt teks melalui encoder visi' },
        status: 'pending'
    },

    render(node, helpers) {
        const el = document.createElement('div');

        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="image" data-port-dir="in" data-port-type="image" data-node-id="${node.id}"></div>
                <div class="ng-port ng-port-in" data-port-name="prompt" data-port-dir="in" data-port-type="string" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body">Image + Text &rarr; Vision</div>
            <div class="ng-node-progress"></div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
