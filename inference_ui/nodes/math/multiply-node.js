// ============================================
// Multiply Node Type (Math category)
// ============================================

NodeRegistry.register('math_multiply', {
    label: 'Multiply',
    category: 'Math',

    ports: [
        { name: 'a', dir: 'in', type: 'numeric', label: 'A', defaultValue: 1 },
        { name: 'b', dir: 'in', type: 'numeric', label: 'B', defaultValue: 1 },
        { name: 'out', dir: 'out', type: 'numeric', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Multiply',
        menuTag: { en: 'Math', ko: '곱셈', ja: '数学', zh: '数学', fr: 'Math', de: 'Mathe', es: 'Matemáticas', it: 'Matematica', pt: 'Matemática', nl: 'Wiskunde', ru: 'Математика', ar: 'رياضيات', hi: 'गणित', tr: 'Matematik', pl: 'Matematyka', cs: 'Matematika', sv: 'Matematik', da: 'Matematik', no: 'Matematikk', fi: 'Matematiikka', el: 'Μαθηματικά', hu: 'Matematika', ro: 'Matematică', uk: 'Математика', vi: 'Toán học', th: 'คณิตศาสตร์', id: 'Matematika' },
        description: { en: 'Multiply two values (A * B)', ko: '두 값을 곱하기 (A × B)', ja: '2つの値を乗算 (A × B)', zh: '两个值相乘 (A × B)', fr: 'Multiplier deux valeurs (A × B)', de: 'Zwei Werte multiplizieren (A × B)', es: 'Multiplicar dos valores (A × B)', it: 'Moltiplica due valori (A × B)', pt: 'Multiplicar dois valores (A × B)', nl: 'Twee waarden vermenigvuldigen (A × B)', ru: 'Умножить два значения (A × B)', ar: 'ضرب قيمتين (A × B)', hi: 'दो मानों को गुणा करें (A × B)', tr: 'İki değeri çarp (A × B)', pl: 'Pomnóż dwie wartości (A × B)', cs: 'Vynásobit dvě hodnoty (A × B)', sv: 'Multiplicera två värden (A × B)', da: 'Multiplicer to værdier (A × B)', no: 'Multipliser to verdier (A × B)', fi: 'Kerro kaksi arvoa (A × B)', el: 'Πολλαπλασιασμός δύο τιμών (A × B)', hu: 'Két érték szorzása (A × B)', ro: 'Înmulțirea a două valori (A × B)', uk: 'Помножити два значення (A × B)', vi: 'Nhân hai giá trị (A × B)', th: 'คูณสองค่า (A × B)', id: 'Kalikan dua nilai (A × B)' },
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
