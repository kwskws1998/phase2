// ============================================
// Sqrt Node Type (Math category)
// ============================================

NodeRegistry.register('math_sqrt', {
    label: 'Sqrt',
    category: 'Math',

    ports: [
        { name: 'value', dir: 'in', type: 'numeric', label: 'Value', defaultValue: 4 },
        { name: 'out', dir: 'out', type: 'numeric', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Sqrt',
        menuTag: { en: 'Math', ko: '제곱근', ja: '数学', zh: '数学', fr: 'Math', de: 'Mathe', es: 'Matemáticas', it: 'Matematica', pt: 'Matemática', nl: 'Wiskunde', ru: 'Математика', ar: 'رياضيات', hi: 'गणित', tr: 'Matematik', pl: 'Matematyka', cs: 'Matematika', sv: 'Matematik', da: 'Matematik', no: 'Matematikk', fi: 'Matematiikka', el: 'Μαθηματικά', hu: 'Matematika', ro: 'Matematică', uk: 'Математика', vi: 'Toán học', th: 'คณิตศาสตร์', id: 'Matematika' },
        description: { en: 'Square root of a value', ko: '값의 제곱근 계산', ja: '値の平方根', zh: '值的平方根', fr: 'Racine carrée de la valeur', de: 'Quadratwurzel eines Werts', es: 'Raíz cuadrada de un valor', it: 'Radice quadrata di un valore', pt: 'Raiz quadrada de um valor', nl: 'Vierkantswortel van een waarde', ru: 'Квадратный корень значения', ar: 'الجذر التربيعي لقيمة', hi: 'एक मान का वर्गमूल', tr: 'Bir değerin karekökü', pl: 'Pierwiastek kwadratowy wartości', cs: 'Odmocnina hodnoty', sv: 'Kvadratroten av ett värde', da: 'Kvadratrod af en værdi', no: 'Kvadratrot av en verdi', fi: 'Arvon neliöjuuri', el: 'Τετραγωνική ρίζα μιας τιμής', hu: 'Egy érték négyzetgyöke', ro: 'Rădăcina pătrată a unei valori', uk: 'Квадратний корінь значення', vi: 'Căn bậc hai của giá trị', th: 'รากที่สองของค่า', id: 'Akar kuadrat dari sebuah nilai' },
        status: 'pending',
        portValues: { value: 4 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'value', label: 'Value', type: 'numeric', defaultValue: 4 }
        ], 'numeric');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
