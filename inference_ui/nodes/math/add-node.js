// ============================================
// Add Node Type (Math category)
// ============================================

NodeRegistry.register('math_add', {
    label: 'Add',
    category: 'Math',

    ports: [
        { name: 'a', dir: 'in', type: 'addable', label: 'A', defaultValue: 0 },
        { name: 'b', dir: 'in', type: 'addable', label: 'B', defaultValue: 0 },
        { name: 'out', dir: 'out', type: 'addable', label: 'Result' }
    ],

    resolveOutputType(inputTypes) {
        if (inputTypes.length === 0) return 'addable';
        if (inputTypes.some(t => t === 'string')) return 'string';
        const allNumeric = inputTypes.every(t =>
            PortTypes._groups['numeric']?.has(t) || t === 'numeric'
        );
        if (allNumeric) return 'numeric';
        return 'addable';
    },

    defaultConfig: {
        title: 'Add',
        menuTag: { en: 'Math', ko: '덧셈', ja: '数学', zh: '数学', fr: 'Math', de: 'Mathe', es: 'Matemáticas', it: 'Matematica', pt: 'Matemática', nl: 'Wiskunde', ru: 'Математика', ar: 'رياضيات', hi: 'गणित', tr: 'Matematik', pl: 'Matematyka', cs: 'Matematika', sv: 'Matematik', da: 'Matematik', no: 'Matematikk', fi: 'Matematiikka', el: 'Μαθηματικά', hu: 'Matematika', ro: 'Matematică', uk: 'Математика', vi: 'Toán học', th: 'คณิตศาสตร์', id: 'Matematika' },
        description: { en: 'Add two values (A + B)', ko: '두 값을 더하기 (A + B)', ja: '2つの値を加算 (A + B)', zh: '两个值相加 (A + B)', fr: 'Additionner deux valeurs (A + B)', de: 'Zwei Werte addieren (A + B)', es: 'Sumar dos valores (A + B)', it: 'Somma due valori (A + B)', pt: 'Somar dois valores (A + B)', nl: 'Twee waarden optellen (A + B)', ru: 'Сложить два значения (A + B)', ar: 'جمع قيمتين (A + B)', hi: 'दो मानों को जोड़ें (A + B)', tr: 'İki değeri topla (A + B)', pl: 'Dodaj dwie wartości (A + B)', cs: 'Sečíst dvě hodnoty (A + B)', sv: 'Addera två värden (A + B)', da: 'Læg to værdier sammen (A + B)', no: 'Legg sammen to verdier (A + B)', fi: 'Lisää kaksi arvoa (A + B)', el: 'Πρόσθεση δύο τιμών (A + B)', hu: 'Két érték összeadása (A + B)', ro: 'Adunarea a două valori (A + B)', uk: 'Додати два значення (A + B)', vi: 'Cộng hai giá trị (A + B)', th: 'บวกสองค่า (A + B)', id: 'Jumlahkan dua nilai (A + B)' },
        status: 'pending',
        portValues: { a: 0, b: 0 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'a', label: 'A', type: 'addable', defaultValue: 0 },
            { name: 'b', label: 'B', type: 'addable', defaultValue: 0 }
        ], 'addable');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
