// ============================================
// PubMed Search Tool Node
// ============================================

NodeRegistry.register('pubmed', {
    label: 'PubMed Search',
    category: 'Tool',
    allowRef: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' },
        { name: 'out', dir: 'out', type: 'any' }
    ],

    defaultConfig: {
        title: 'PubMed Search',
        tool: 'pubmed_search',
        menuTag: {
            en: 'Papers', ko: '논문 검색', ja: '論文', zh: '论文',
            fr: 'Articles', de: 'Artikel', es: 'Artículos', it: 'Articoli',
            pt: 'Artigos', nl: 'Artikelen', ru: 'Статьи', ar: 'أبحاث',
            hi: 'शोधपत्र', tr: 'Makaleler', pl: 'Artykuły', cs: 'Články',
            sv: 'Artiklar', da: 'Artikler', no: 'Artikler', fi: 'Julkaisut',
            el: 'Άρθρα', hu: 'Cikkek', ro: 'Articole', uk: 'Статті',
            vi: 'Bài báo', th: 'บทความ', id: 'Artikel'
        },
        description: {
            en: 'Search PubMed for research papers and publications',
            ko: 'PubMed에서 연구 논문 및 출판물 검색',
            ja: 'PubMedで研究論文や出版物を検索',
            zh: '在PubMed中搜索研究论文和出版物',
            fr: 'Rechercher des articles de recherche et des publications sur PubMed',
            de: 'PubMed nach Forschungsartikeln und Publikationen durchsuchen',
            es: 'Buscar artículos de investigación y publicaciones en PubMed',
            it: 'Cercare articoli di ricerca e pubblicazioni su PubMed',
            pt: 'Pesquisar artigos de pesquisa e publicações no PubMed',
            nl: 'Zoek naar onderzoeksartikelen en publicaties op PubMed',
            ru: 'Поиск научных статей и публикаций в PubMed',
            ar: 'البحث في PubMed عن الأبحاث والمنشورات العلمية',
            hi: 'PubMed पर शोध पत्र और प्रकाशन खोजें',
            tr: 'PubMed\'de araştırma makaleleri ve yayınları arayın',
            pl: 'Wyszukaj artykuły naukowe i publikacje w PubMed',
            cs: 'Vyhledávejte výzkumné články a publikace na PubMed',
            sv: 'Sök efter forskningsartiklar och publikationer på PubMed',
            da: 'Søg efter forskningsartikler og publikationer på PubMed',
            no: 'Søk etter forskningsartikler og publikasjoner på PubMed',
            fi: 'Etsi tutkimusartikkeleita ja julkaisuja PubMedistä',
            el: 'Αναζήτηση ερευνητικών άρθρων και δημοσιεύσεων στο PubMed',
            hu: 'Kutatási cikkek és publikációk keresése a PubMed-ben',
            ro: 'Căutați articole de cercetare și publicații pe PubMed',
            uk: 'Пошук наукових статей та публікацій у PubMed',
            vi: 'Tìm kiếm bài báo nghiên cứu và ấn phẩm trên PubMed',
            th: 'ค้นหาบทความวิจัยและสิ่งพิมพ์บน PubMed',
            id: 'Cari artikel penelitian dan publikasi di PubMed'
        },
        status: 'pending',
        stepNum: ''
    },

    render(node, helpers) {
        const el = document.createElement('div');
        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-num">${helpers.escapeHtml(node.stepNum)}</span>
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body">${helpers.escapeHtml(node.tool)}</div>
            <div class="ng-node-progress"></div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
        `;
        return el;
    },

    getDragHandle(el) { return el.querySelector('.ng-node-header'); }
});
