// ============================================
// Subtract Node Type (Math category)
// ============================================

NodeRegistry.register('math_subtract', {
    label: 'Subtract',
    category: 'Math',

    ports: [
        { name: 'a', dir: 'in', type: 'numeric', label: 'A', defaultValue: 0 },
        { name: 'b', dir: 'in', type: 'numeric', label: 'B', defaultValue: 0 },
        { name: 'out', dir: 'out', type: 'numeric', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Subtract',
        menuTag: { en: 'Math', ko: '뺄셈', ja: '数学', zh: '数学', fr: 'Math', de: 'Mathe', es: 'Matemáticas', it: 'Matematica', pt: 'Matemática', nl: 'Wiskunde', ru: 'Математика', ar: 'رياضيات', hi: 'गणित', tr: 'Matematik', pl: 'Matematyka', cs: 'Matematika', sv: 'Matematik', da: 'Matematik', no: 'Matematikk', fi: 'Matematiikka', el: 'Μαθηματικά', hu: 'Matematika', ro: 'Matematică', uk: 'Математика', vi: 'Toán học', th: 'คณิตศาสตร์', id: 'Matematika' },
        description: { en: 'Subtract two values (A - B)', ko: '두 값을 빼기 (A - B)', ja: '2つの値を減算 (A - B)', zh: '两个值相减 (A - B)', fr: 'Soustraire deux valeurs (A - B)', de: 'Zwei Werte subtrahieren (A - B)', es: 'Restar dos valores (A - B)', it: 'Sottrai due valori (A - B)', pt: 'Subtrair dois valores (A - B)', nl: 'Twee waarden aftrekken (A - B)', ru: 'Вычесть два значения (A - B)', ar: 'طرح قيمتين (A - B)', hi: 'दो मानों को घटाएँ (A - B)', tr: 'İki değeri çıkar (A - B)', pl: 'Odejmij dwie wartości (A - B)', cs: 'Odečíst dvě hodnoty (A - B)', sv: 'Subtrahera två värden (A - B)', da: 'Træk to værdier fra (A - B)', no: 'Trekk fra to verdier (A - B)', fi: 'Vähennä kaksi arvoa (A - B)', el: 'Αφαίρεση δύο τιμών (A - B)', hu: 'Két érték kivonása (A - B)', ro: 'Scăderea a două valori (A - B)', uk: 'Відняти два значення (A - B)', vi: 'Trừ hai giá trị (A - B)', th: 'ลบสองค่า (A - B)', id: 'Kurangi dua nilai (A - B)' },
        status: 'pending',
        portValues: { a: 0, b: 0 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'a', label: 'A', type: 'numeric', defaultValue: 0 },
            { name: 'b', label: 'B', type: 'numeric', defaultValue: 0 }
        ], 'numeric');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
