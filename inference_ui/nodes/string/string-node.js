// ============================================
// String Node Type (Input category)
// ============================================

NodeRegistry.register('string', {
    label: 'String',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'string', label: 'Value' }
    ],

    defaultConfig: {
        title: 'String',
        menuTag: {
            en: 'Text', ko: '텍스트',
            ja: 'テキスト', zh: '文本',
            fr: 'Texte', de: 'Text', es: 'Texto', it: 'Testo', pt: 'Texto',
            nl: 'Tekst', ru: 'Текст', ar: 'نص', hi: 'पाठ',
            tr: 'Metin', pl: 'Tekst', cs: 'Text', sv: 'Text',
            da: 'Tekst', no: 'Tekst', fi: 'Teksti', el: 'Κείμενο',
            hu: 'Szöveg', ro: 'Text', uk: 'Текст',
            vi: 'Văn bản', th: 'ข้อความ', id: 'Teks'
        },
        description: {
            en: 'Text string input value for passing text data',
            ko: '텍스트 문자열 입력값',
            ja: 'テキストデータを渡すための文字列入力値',
            zh: '用于传递文本数据的字符串输入值',
            fr: 'Chaîne de texte en entrée pour transmettre des données textuelles',
            de: 'Texteingabewert zur Übergabe von Textdaten',
            es: 'Valor de entrada de cadena de texto para pasar datos de texto',
            it: 'Valore di input stringa di testo per passare dati testuali',
            pt: 'Valor de entrada de string de texto para passar dados de texto',
            nl: 'Tekstinvoerwaarde voor het doorgeven van tekstgegevens',
            ru: 'Строковое входное значение для передачи текстовых данных',
            ar: 'قيمة إدخال نصية لتمرير البيانات النصية',
            hi: 'पाठ डेटा पास करने के लिए टेक्स्ट स्ट्रिंग इनपुट मान',
            tr: 'Metin verilerini iletmek için metin dizesi giriş değeri',
            pl: 'Wartość wejściowa ciągu tekstowego do przekazywania danych tekstowych',
            cs: 'Textový řetězcový vstup pro předávání textových dat',
            sv: 'Textstränginmatning för att skicka textdata',
            da: 'Tekststreng-inputværdi til at sende tekstdata',
            no: 'Tekststrenginngang for å sende tekstdata',
            fi: 'Tekstimerkkijonon syöttöarvo tekstidatan välittämiseen',
            el: 'Τιμή εισόδου συμβολοσειράς κειμένου για τη μετάδοση δεδομένων κειμένου',
            hu: 'Szöveges karakterlánc bemeneti érték szöveges adatok átadásához',
            ro: 'Valoare de intrare șir de text pentru transmiterea datelor text',
            uk: 'Рядкове вхідне значення для передачі текстових даних',
            vi: 'Giá trị đầu vào chuỗi văn bản để truyền dữ liệu văn bản',
            th: 'ค่าอินพุตสตริงข้อความสำหรับส่งข้อมูลข้อความ',
            id: 'Nilai input string teks untuk meneruskan data teks'
        },
        status: 'completed',
        portValues: { out: '' }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const val = (node.portValues && node.portValues.out !== undefined) ? node.portValues.out : '';

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-input-node-body">
                <textarea class="ng-input-node-field ng-port-default ng-interactive" rows="1"
                          placeholder="Enter text..." data-port-ref="out">${helpers.escapeHtml(String(val))}</textarea>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="string" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    afterRender(el, node, helpers) {
        const ta = el.querySelector('textarea');
        if (!ta) return;
        const autoResize = () => {
            ta.style.height = 'auto';
            ta.style.height = ta.scrollHeight + 'px';
            helpers.updateConnections(node);
        };
        ta.addEventListener('input', autoResize);

        let lastWidth = 0;
        const ro = new ResizeObserver(entries => {
            const entry = entries[0];
            if (entry) {
                const w = entry.contentRect.width;
                if (w > 0 && w !== lastWidth) {
                    lastWidth = w;
                    requestAnimationFrame(autoResize);
                }
            }
        });
        ro.observe(el);

        requestAnimationFrame(() => requestAnimationFrame(autoResize));
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
