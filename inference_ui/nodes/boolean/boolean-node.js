// ============================================
// Boolean Node Type (Input category)
// ============================================

NodeRegistry.register('boolean_value', {
    label: 'Boolean',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'boolean', label: 'Value' }
    ],

    defaultConfig: {
        title: 'Boolean',
        menuTag: {
            en: 'Toggle', ko: '참/거짓',
            ja: 'トグル', zh: '开关',
            fr: 'Bascule', de: 'Umschalter', es: 'Interruptor', it: 'Interruttore', pt: 'Alternância',
            nl: 'Schakelaar', ru: 'Переключатель', ar: 'مفتاح تبديل', hi: 'टॉगल',
            tr: 'Anahtar', pl: 'Przełącznik', cs: 'Přepínač', sv: 'Växel',
            da: 'Skifter', no: 'Bryter', fi: 'Kytkin', el: 'Εναλλαγή',
            hu: 'Kapcsoló', ro: 'Comutator', uk: 'Перемикач',
            vi: 'Chuyển đổi', th: 'สลับ', id: 'Sakelar'
        },
        description: {
            en: 'True or false toggle for boolean parameters',
            ko: '참 또는 거짓 불리언 토글',
            ja: 'ブールパラメータ用の真偽トグル',
            zh: '布尔参数的真假切换',
            fr: 'Bascule vrai ou faux pour les paramètres booléens',
            de: 'Wahr/Falsch-Umschalter für boolesche Parameter',
            es: 'Interruptor verdadero o falso para parámetros booleanos',
            it: 'Interruttore vero o falso per parametri booleani',
            pt: 'Alternância verdadeiro ou falso para parâmetros booleanos',
            nl: 'Waar of onwaar schakelaar voor booleaanse parameters',
            ru: 'Переключатель истина/ложь для булевых параметров',
            ar: 'مفتاح تبديل صح أو خطأ للمعلمات المنطقية',
            hi: 'बूलियन पैरामीटर के लिए सही या गलत टॉगल',
            tr: 'Boole parametreleri için doğru veya yanlış anahtarı',
            pl: 'Przełącznik prawda/fałsz dla parametrów logicznych',
            cs: 'Přepínač pravda/nepravda pro booleovské parametry',
            sv: 'Sant eller falskt växel för booleska parametrar',
            da: 'Sand eller falsk skifter for boolske parametre',
            no: 'Sann eller usann bryter for boolske parametere',
            fi: 'Tosi tai epätosi kytkin Boolen parametreille',
            el: 'Εναλλαγή αληθούς ή ψευδούς για παραμέτρους Boolean',
            hu: 'Igaz vagy hamis kapcsoló logikai paraméterekhez',
            ro: 'Comutator adevărat sau fals pentru parametri booleeni',
            uk: 'Перемикач істина/хибність для булевих параметрів',
            vi: 'Chuyển đổi đúng hoặc sai cho tham số boolean',
            th: 'สลับจริงหรือเท็จสำหรับพารามิเตอร์บูลีน',
            id: 'Sakelar benar atau salah untuk parameter boolean'
        },
        status: 'completed',
        portValues: { out: false }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const val = (node.portValues && node.portValues.out !== undefined) ? node.portValues.out : false;

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-input-node-body">
                <div class="ng-boolean-toggle">
                    <input class="ng-interactive" type="checkbox" data-port-ref="out" ${val ? 'checked' : ''}>
                    <span class="ng-boolean-label">${val ? 'true' : 'false'}</span>
                </div>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="boolean" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    afterRender(el, node, helpers) {
        const cb = el.querySelector('input[type="checkbox"]');
        const label = el.querySelector('.ng-boolean-label');
        if (!cb) return;
        cb.addEventListener('change', () => {
            const n = helpers.getNode();
            if (!n) return;
            if (!n.portValues) n.portValues = {};
            n.portValues.out = cb.checked;
            if (label) label.textContent = cb.checked ? 'true' : 'false';
        });
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
