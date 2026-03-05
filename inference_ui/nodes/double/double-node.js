// ============================================
// Double Node Type (Input category)
// ============================================

NodeRegistry.register('double_value', {
    label: 'Double',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'double', label: 'Value' }
    ],

    defaultConfig: {
        title: 'Double',
        menuTag: {
            en: 'Number', ko: '배정밀도',
            ja: '数値', zh: '数字',
            fr: 'Nombre', de: 'Zahl', es: 'Número', it: 'Numero', pt: 'Número',
            nl: 'Nummer', ru: 'Число', ar: 'رقم', hi: 'संख्या',
            tr: 'Sayı', pl: 'Liczba', cs: 'Číslo', sv: 'Nummer',
            da: 'Nummer', no: 'Nummer', fi: 'Numero', el: 'Αριθμός',
            hu: 'Szám', ro: 'Număr', uk: 'Число',
            vi: 'Số', th: 'ตัวเลข', id: 'Angka'
        },
        description: {
            en: 'High-precision 64-bit floating-point input',
            ko: '64비트 고정밀 부동소수점 입력',
            ja: '高精度64ビット浮動小数点入力',
            zh: '高精度64位浮点数输入',
            fr: 'Entrée à virgule flottante 64 bits haute précision',
            de: 'Hochpräzise 64-Bit-Gleitkommaeingabe',
            es: 'Entrada de punto flotante de 64 bits de alta precisión',
            it: 'Input a virgola mobile a 64 bit ad alta precisione',
            pt: 'Entrada de ponto flutuante de 64 bits de alta precisão',
            nl: 'Hoge-precisie 64-bit drijvende-komma invoer',
            ru: 'Высокоточный 64-битный ввод с плавающей запятой',
            ar: 'إدخال عدد عشري 64 بت عالي الدقة',
            hi: 'उच्च-सटीकता 64-बिट फ्लोटिंग-पॉइंट इनपुट',
            tr: 'Yüksek hassasiyetli 64-bit kayan noktalı giriş',
            pl: 'Wejście zmiennoprzecinkowe 64-bitowe o wysokiej precyzji',
            cs: 'Vysoce přesný 64bitový vstup s plovoucí desetinnou čárkou',
            sv: '64-bitars flyttalsinmatning med hög precision',
            da: '64-bit flydende komma-indgang med høj præcision',
            no: 'Høypresisjons 64-bits flyttallsinngang',
            fi: 'Korkean tarkkuuden 64-bittinen liukulukusyöte',
            el: 'Είσοδος κινητής υποδιαστολής 64 bit υψηλής ακρίβειας',
            hu: 'Nagy pontosságú 64 bites lebegőpontos bemenet',
            ro: 'Intrare cu virgulă mobilă pe 64 de biți de înaltă precizie',
            uk: 'Високоточне 64-бітне введення з плаваючою комою',
            vi: 'Đầu vào dấu phẩy động 64-bit độ chính xác cao',
            th: 'อินพุตเลขทศนิยม 64 บิตความแม่นยำสูง',
            id: 'Input titik mengambang 64-bit presisi tinggi'
        },
        status: 'completed',
        portValues: { out: 0.0 }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const val = (node.portValues && node.portValues.out !== undefined) ? node.portValues.out : 0.0;

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-input-node-body">
                <input class="ng-input-node-field ng-port-default ng-interactive" type="number" step="any" value="${val}"
                       data-port-ref="out">
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="double" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
