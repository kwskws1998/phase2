// ============================================
// Log Node Type (Math category)
// ============================================

NodeRegistry.register('math_log', {
    label: 'Log',
    category: 'Math',

    ports: [
        { name: 'value', dir: 'in', type: 'numeric', label: 'Value', defaultValue: 1 },
        { name: 'base', dir: 'in', type: 'numeric', label: 'Base', defaultValue: 2.718 },
        { name: 'out', dir: 'out', type: 'numeric', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Log',
        menuTag: { en: 'Math', ko: '로그', ja: '数学', zh: '数学', fr: 'Math', de: 'Mathe', es: 'Matemáticas', it: 'Matematica', pt: 'Matemática', nl: 'Wiskunde', ru: 'Математика', ar: 'رياضيات', hi: 'गणित', tr: 'Matematik', pl: 'Matematyka', cs: 'Matematika', sv: 'Matematik', da: 'Matematik', no: 'Matematikk', fi: 'Matematiikka', el: 'Μαθηματικά', hu: 'Matematika', ro: 'Matematică', uk: 'Математика', vi: 'Toán học', th: 'คณิตศาสตร์', id: 'Matematika' },
        description: { en: 'Logarithm with custom base', ko: '사용자 지정 밑의 로그 계산', ja: 'カスタム底の対数', zh: '自定义底数的对数', fr: 'Logarithme avec base personnalisée', de: 'Logarithmus mit benutzerdefinierter Basis', es: 'Logaritmo con base personalizada', it: 'Logaritmo con base personalizzata', pt: 'Logaritmo com base personalizada', nl: 'Logaritme met aangepaste basis', ru: 'Логарифм с настраиваемым основанием', ar: 'لوغاريتم بأساس مخصص', hi: 'कस्टम आधार के साथ लघुगणक', tr: 'Özel tabanlı logaritma', pl: 'Logarytm z niestandardową podstawą', cs: 'Logaritmus s vlastním základem', sv: 'Logaritm med anpassad bas', da: 'Logaritme med brugerdefineret base', no: 'Logaritme med egendefinert base', fi: 'Logaritmi mukautetulla kantaluvulla', el: 'Λογάριθμος με προσαρμοσμένη βάση', hu: 'Logaritmus egyéni alappal', ro: 'Logaritm cu bază personalizată', uk: 'Логарифм з довільною основою', vi: 'Logarit với cơ số tùy chỉnh', th: 'ลอการิทึมกับฐานที่กำหนดเอง', id: 'Logaritma dengan basis kustom' },
        status: 'pending',
        portValues: { value: 1, base: 2.718 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'value', label: 'Value', type: 'numeric', defaultValue: 1 },
            { name: 'base', label: 'Base', type: 'numeric', defaultValue: 2.718 }
        ], 'numeric');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
