// ============================================
// Integer Node Type (Input category)
// ============================================

NodeRegistry.register('integer', {
    label: 'Integer',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'int', label: 'Value' }
    ],

    defaultConfig: {
        title: 'Integer',
        menuTag: {
            en: 'Number', ko: '정수',
            ja: '数値', zh: '数字',
            fr: 'Nombre', de: 'Zahl', es: 'Número', it: 'Numero', pt: 'Número',
            nl: 'Nummer', ru: 'Число', ar: 'رقم', hi: 'संख्या',
            tr: 'Sayı', pl: 'Liczba', cs: 'Číslo', sv: 'Nummer',
            da: 'Nummer', no: 'Nummer', fi: 'Numero', el: 'Αριθμός',
            hu: 'Szám', ro: 'Număr', uk: 'Число',
            vi: 'Số', th: 'ตัวเลข', id: 'Angka'
        },
        description: {
            en: 'Whole number input for integer parameters',
            ko: '정수형 숫자 입력',
            ja: '整数パラメータ用の整数入力',
            zh: '用于整数参数的整数输入',
            fr: 'Entrée de nombre entier pour les paramètres entiers',
            de: 'Ganzzahleingabe für ganzzahlige Parameter',
            es: 'Entrada de número entero para parámetros enteros',
            it: 'Input di numero intero per parametri interi',
            pt: 'Entrada de número inteiro para parâmetros inteiros',
            nl: 'Geheel getal invoer voor integer parameters',
            ru: 'Ввод целого числа для целочисленных параметров',
            ar: 'إدخال عدد صحيح لمعلمات الأعداد الصحيحة',
            hi: 'पूर्णांक पैरामीटर के लिए पूर्ण संख्या इनपुट',
            tr: 'Tam sayı parametreleri için tam sayı girişi',
            pl: 'Wejście liczby całkowitej dla parametrów całkowitych',
            cs: 'Vstup celého čísla pro celočíselné parametry',
            sv: 'Heltalsinmatning för heltalsparametrar',
            da: 'Heltalsindtastning for heltalsparametre',
            no: 'Heltallsinngang for heltallsparametere',
            fi: 'Kokonaislukusyöte kokonaislukuparametreille',
            el: 'Είσοδος ακέραιου αριθμού για ακέραιες παραμέτρους',
            hu: 'Egész szám bemenet egész paraméterekhez',
            ro: 'Intrare de număr întreg pentru parametri întregi',
            uk: 'Введення цілого числа для цілочисельних параметрів',
            vi: 'Đầu vào số nguyên cho các tham số nguyên',
            th: 'อินพุตจำนวนเต็มสำหรับพารามิเตอร์จำนวนเต็ม',
            id: 'Input bilangan bulat untuk parameter bilangan bulat'
        },
        status: 'completed',
        portValues: { out: 0 }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const val = (node.portValues && node.portValues.out !== undefined) ? node.portValues.out : 0;

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-input-node-body">
                <input class="ng-input-node-field ng-port-default ng-interactive" type="number" step="1" value="${val}"
                       data-port-ref="out">
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="int" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
