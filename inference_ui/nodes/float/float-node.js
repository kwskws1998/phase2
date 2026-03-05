// ============================================
// Float Node Type (Input category)
// ============================================

NodeRegistry.register('float', {
    label: 'Float',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'float', label: 'Value' }
    ],

    defaultConfig: {
        title: 'Float',
        menuTag: {
            en: 'Number', ko: '실수',
            ja: '数値', zh: '数字',
            fr: 'Nombre', de: 'Zahl', es: 'Número', it: 'Numero', pt: 'Número',
            nl: 'Nummer', ru: 'Число', ar: 'رقم', hi: 'संख्या',
            tr: 'Sayı', pl: 'Liczba', cs: 'Číslo', sv: 'Nummer',
            da: 'Nummer', no: 'Nummer', fi: 'Numero', el: 'Αριθμός',
            hu: 'Szám', ro: 'Număr', uk: 'Число',
            vi: 'Số', th: 'ตัวเลข', id: 'Angka'
        },
        description: {
            en: 'Floating-point number input for decimal values',
            ko: '부동소수점 숫자 입력',
            ja: '小数値用の浮動小数点数入力',
            zh: '用于小数值的浮点数输入',
            fr: 'Entrée de nombre à virgule flottante pour les valeurs décimales',
            de: 'Gleitkommazahleingabe für Dezimalwerte',
            es: 'Entrada de número de punto flotante para valores decimales',
            it: 'Input di numero a virgola mobile per valori decimali',
            pt: 'Entrada de número de ponto flutuante para valores decimais',
            nl: 'Drijvende-kommagetal invoer voor decimale waarden',
            ru: 'Ввод числа с плавающей запятой для десятичных значений',
            ar: 'إدخال عدد عشري للقيم العشرية',
            hi: 'दशमलव मानों के लिए फ्लोटिंग-पॉइंट संख्या इनपुट',
            tr: 'Ondalık değerler için kayan noktalı sayı girişi',
            pl: 'Wejście liczby zmiennoprzecinkowej dla wartości dziesiętnych',
            cs: 'Vstup čísla s plovoucí desetinnou čárkou pro desetinné hodnoty',
            sv: 'Flyttalsinmatning för decimalvärden',
            da: 'Flydende komma-talindtastning for decimalværdier',
            no: 'Flyttallsinngang for desimalverdier',
            fi: 'Liukulukusyöte desimaaliarvoille',
            el: 'Είσοδος αριθμού κινητής υποδιαστολής για δεκαδικές τιμές',
            hu: 'Lebegőpontos szám bemenet tizedes értékekhez',
            ro: 'Intrare de număr cu virgulă mobilă pentru valori zecimale',
            uk: 'Введення числа з плаваючою комою для десяткових значень',
            vi: 'Đầu vào số dấu phẩy động cho giá trị thập phân',
            th: 'อินพุตเลขทศนิยมสำหรับค่าทศนิยม',
            id: 'Input angka titik mengambang untuk nilai desimal'
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
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="float" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
