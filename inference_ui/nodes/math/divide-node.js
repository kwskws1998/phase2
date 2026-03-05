// ============================================
// Divide Node Type (Math category)
// ============================================

NodeRegistry.register('math_divide', {
    label: 'Divide',
    category: 'Math',

    ports: [
        { name: 'a', dir: 'in', type: 'numeric', label: 'A', defaultValue: 1 },
        { name: 'b', dir: 'in', type: 'numeric', label: 'B', defaultValue: 1 },
        { name: 'out', dir: 'out', type: 'numeric', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Divide',
        menuTag: { en: 'Math', ko: '나눗셈', ja: '数学', zh: '数学', fr: 'Math', de: 'Mathe', es: 'Matemáticas', it: 'Matematica', pt: 'Matemática', nl: 'Wiskunde', ru: 'Математика', ar: 'رياضيات', hi: 'गणित', tr: 'Matematik', pl: 'Matematyka', cs: 'Matematika', sv: 'Matematik', da: 'Matematik', no: 'Matematikk', fi: 'Matematiikka', el: 'Μαθηματικά', hu: 'Matematika', ro: 'Matematică', uk: 'Математика', vi: 'Toán học', th: 'คณิตศาสตร์', id: 'Matematika' },
        description: { en: 'Divide two values (A / B)', ko: '두 값을 나누기 (A ÷ B)', ja: '2つの値を除算 (A ÷ B)', zh: '两个值相除 (A ÷ B)', fr: 'Diviser deux valeurs (A ÷ B)', de: 'Zwei Werte dividieren (A ÷ B)', es: 'Dividir dos valores (A ÷ B)', it: 'Dividere due valori (A ÷ B)', pt: 'Dividir dois valores (A ÷ B)', nl: 'Twee waarden delen (A ÷ B)', ru: 'Разделить два значения (A ÷ B)', ar: 'قسمة قيمتين (A ÷ B)', hi: 'दो मानों को विभाजित करें (A ÷ B)', tr: 'İki değeri böl (A ÷ B)', pl: 'Podziel dwie wartości (A ÷ B)', cs: 'Vydělit dvě hodnoty (A ÷ B)', sv: 'Dividera två värden (A ÷ B)', da: 'Divider to værdier (A ÷ B)', no: 'Divider to verdier (A ÷ B)', fi: 'Jaa kaksi arvoa (A ÷ B)', el: 'Διαίρεση δύο τιμών (A ÷ B)', hu: 'Két érték osztása (A ÷ B)', ro: 'Împărțirea a două valori (A ÷ B)', uk: 'Поділити два значення (A ÷ B)', vi: 'Chia hai giá trị (A ÷ B)', th: 'หารสองค่า (A ÷ B)', id: 'Bagi dua nilai (A ÷ B)' },
        status: 'pending',
        portValues: { a: 1, b: 1 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'a', label: 'A', type: 'numeric', defaultValue: 1 },
            { name: 'b', label: 'B', type: 'numeric', defaultValue: 1 }
        ], 'numeric');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
