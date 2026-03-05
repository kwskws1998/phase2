// ============================================
// Code Gen Tool Node
// ============================================

NodeRegistry.register('codegen', {
    label: 'Code Gen',
    category: 'Tool',
    allowRef: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' },
        { name: 'out', dir: 'out', type: 'any' }
    ],

    defaultConfig: {
        title: 'Code Gen',
        tool: 'code_gen',
        menuTag: {
            en: 'Code', ko: '코드', ja: 'コード', zh: '代码',
            fr: 'Code', de: 'Code', es: 'Código', it: 'Codice',
            pt: 'Código', nl: 'Code', ru: 'Код', ar: 'كود',
            hi: 'कोड', tr: 'Kod', pl: 'Kod', cs: 'Kód',
            sv: 'Kod', da: 'Kode', no: 'Kode', fi: 'Koodi',
            el: 'Κώδικας', hu: 'Kód', ro: 'Cod', uk: 'Код',
            vi: 'Mã', th: 'โค้ด', id: 'Kode'
        },
        description: {
            en: 'Generate Python or R code from natural language',
            ko: '자연어로부터 Python 또는 R 코드 생성',
            ja: '自然言語からPythonまたはRコードを生成',
            zh: '从自然语言生成Python或R代码',
            fr: 'Générer du code Python ou R à partir du langage naturel',
            de: 'Python- oder R-Code aus natürlicher Sprache generieren',
            es: 'Generar código Python o R a partir de lenguaje natural',
            it: 'Generare codice Python o R dal linguaggio naturale',
            pt: 'Gerar código Python ou R a partir de linguagem natural',
            nl: 'Python- of R-code genereren uit natuurlijke taal',
            ru: 'Генерация кода Python или R из естественного языка',
            ar: 'توليد كود Python أو R من اللغة الطبيعية',
            hi: 'प्राकृतिक भाषा से Python या R कोड उत्पन्न करें',
            tr: 'Doğal dilden Python veya R kodu oluşturun',
            pl: 'Generuj kod Python lub R z języka naturalnego',
            cs: 'Generování kódu Python nebo R z přirozeného jazyka',
            sv: 'Generera Python- eller R-kod från naturligt språk',
            da: 'Generer Python- eller R-kode fra naturligt sprog',
            no: 'Generer Python- eller R-kode fra naturlig språk',
            fi: 'Luo Python- tai R-koodia luonnollisesta kielestä',
            el: 'Δημιουργία κώδικα Python ή R από φυσική γλώσσα',
            hu: 'Python vagy R kód generálása természetes nyelvből',
            ro: 'Generați cod Python sau R din limbaj natural',
            uk: 'Генерація коду Python або R з природної мови',
            vi: 'Tạo mã Python hoặc R từ ngôn ngữ tự nhiên',
            th: 'สร้างโค้ด Python หรือ R จากภาษาธรรมชาติ',
            id: 'Hasilkan kode Python atau R dari bahasa alami'
        },
        status: 'pending',
        stepNum: ''
    },

    render(node, helpers) {
        const el = document.createElement('div');
        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-num">${helpers.escapeHtml(node.stepNum)}</span>
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body">${helpers.escapeHtml(node.tool)}</div>
            <div class="ng-node-progress"></div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
        `;
        return el;
    },

    getDragHandle(el) { return el.querySelector('.ng-node-header'); }
});
