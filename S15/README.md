**The assignment was to rewrite the whole code covered in the class in Pytorch-Lightning.Train the model for 10 epochs and achieve the loss of less than 4. 

In the assignment I had run total 20 EPOCHS and for the 10th EPOCH I was able to get ... loss. 

COPY of the logs is as follows:

***Max length of source sentence: 309
Max length of target sentence: 274***
Processing Epoch 00: 100% 4850/4850 [04:44<00:00, 17.06it/s, loss=6.510]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: I was tormented by the contrast between my idea and my handiwork: in each case I had imagined something which I was quite powerless to realise."
    TARGET: Anzi soffrivo per il contrasto che vi era fra l'ideale e l'opera e mi sentivo impotente a dar forma alle immagini della mia mente.
    PREDICTED: Non , e , e , e , e , e , e .
--------------------------------------------------------------------------------
    SOURCE: Now it's settled...'
    TARGET: Ormai è concluso....
    PREDICTED: E che è un ’ ic .
--------------------------------------------------------------------------------
Processing Epoch 01: 100% 4850/4850 [04:47<00:00, 16.84it/s, loss=5.605]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: Grimm has a fable called "The Man Without a Shadow" – about a man who lost his shadow as a punishment for something or other.
    TARGET: C’è una favola di Grimm: l’uomo senza ombra, l’uomo privato dell’ombra. E questo gli è dato in castigo di qualcosa.
    PREDICTED: — , — disse , — e , — e , — e .
--------------------------------------------------------------------------------
    SOURCE: 'Fürst Shcherbatsky sammt Gemahlin und Tochter,' [Prince Shcherbatsky with his wife and daughter.] by the premises they occupied, by their name, and by the people they were acquainted with, at once crystallized into their definite and preordained place.
    TARGET: Fürst Šcerbackij sammt Gemahlin und Tochter per il nome e per l’appartamento che occupavano e per gli amici che avevano trovato, si cristallizzarono nel loro posto definito e ad essi destinato.
    PREDICTED: — , , , , e , , e , e , e , e , e , e e .
--------------------------------------------------------------------------------
Processing Epoch 02: 100% 4850/4850 [04:56<00:00, 16.37it/s, loss=3.982]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: 'Oh, nothing –' answered Oblonsky. 'We'll talk it over later on.
    TARGET: — No, nulla — rispose Oblonskij. — Ne riparleremo.
    PREDICTED: — Oh , no — disse Stepan Arkad ’ ic .
--------------------------------------------------------------------------------
    SOURCE: I had, God knows, more sincerity than knowledge in all the methods I took for this poor creature’s instruction, and must acknowledge, what I believe all that act upon the same principle will find, that in laying things open to him, I really informed and instructed myself in many things that either I did not know or had not fully considered before, but which occurred naturally to my mind upon searching into them, for the information of this poor savage; and I had more affection in my inquiry after things upon this occasion than ever I felt before: so that, whether this poor wild wretch was better for me or no, I had great reason to be thankful that ever he came to me; my grief sat lighter, upon me; my habitation grew comfortable to me beyond measure: and when I reflected that in this solitary life which I have been confined to, I had not only been moved to look up to heaven myself, and to seek the Hand that had brought me here, but was now to be made an instrument, under Providence, to save the life, and, for aught I knew, the soul of a poor savage, and bring him to the true knowledge of religion and of the Christian doctrine, that he might know Christ Jesus, in whom is life eternal; I say, when I reflected upon all these things, a secret joy ran through every part of My soul, and I frequently rejoiced that ever I was brought to this place, which I had so often thought the most dreadful of all afflictions that could possibly have befallen me.
    TARGET: In somma, sia o no divenuto migliore per opera mia quello sfortunato, certo ho grande motivo di ringraziare la celeste provvidenza che me lo inviò. I miei cordogli da quell’istante divennero più leggieri; la mia abitazione mi si rese oltremodo cara; e quando pensava che questo solitario confine mi fu non solo un impulso a volgere gli sguardi al cielo io medesimo e a cercare con affetto la mano che mi vi aveva condotto, ma era per rendermi con l’aiuto di Dio uno stromento alto a fare salva la vita e, a quanto sembrommi, l’anima di un povero selvaggio ed a condurlo su la via della religione e degl’insegnamenti della cristiana dottrina e dell’adorazione di Gesù Cristo in cui è la vita eterna: quando io pensava a tutto ciò, una segreta gioia comprendeva ogni parte della mia anima; e una tale idea frequentemente mi è stata di consolazione sino al termine del mio esilio in questo luogo: esilio ch’io aveva si spesso riguardato come la più spaventosa fra quante sventura avessero mai potuto avvenirmi.
    PREDICTED: Io mi era stato stato più di questo , che mi per , e io mi per me , e che mi , e che mi , e che mi , e che mi , e che mi , e che mi , e che mi , e che non mi , e che mi , e che mi , e che mi , e che mi , e che non mi mai più che mi , e non mi mai più che mi , e che non mai più che mi , e che mi , e che non mi mai più che non mi mai mai più che non mi mai che non mi mai mai mai che mi mai mai mai , e che mi , e che mi , e che mi , e che mi , e che mi , e che mi , e che mi mai , e che mi , e che mi , e che mi mai , e che mi , e che mi , e che mi , e che mi , e che mi , e che mi , e che mi , e che mi , e che mi , e che mi , e non , e che mi , e che mi , e non , e che mi mi mi mi mi mi mi , e che mi , e non , e che mi , e che mi , e non mi , e che mi , e che mi , e che mi mi mi mi , e , e ,
--------------------------------------------------------------------------------
Processing Epoch 03: 100% 4850/4850 [04:49<00:00, 16.74it/s, loss=5.515]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: He lay awake half the night considering the details necessary for carrying his thought into effect.
    TARGET: Non dormì per metà della notte, pensando ai particolari dell’esecuzione della sua idea.
    PREDICTED: Egli si era già già in una settimana in cui si era avvicinato .
--------------------------------------------------------------------------------
    SOURCE: Trying not to hurry and not to get excited, Levin gave the names of the doctor and of the midwife, explained why the opium was wanted and tried to persuade the dispenser to let him have it.
    TARGET: Cercando di non avere fretta e di non accalorarsi, fatto il nome del dottore e quello della levatrice, e spiegato perché serviva l’oppio, Levin cominciò a persuaderlo.
    PREDICTED: Non si poteva andare e non si , Levin , e Levin , e la principessa , senza , si poteva e si poteva fare il dottore , e si poteva fare .
--------------------------------------------------------------------------------
Processing Epoch 04: 100% 4850/4850 [04:49<00:00, 16.78it/s, loss=4.426]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: The river up to Sonning winds in and out through many islands, and is very placid, hushed, and lonely.
    TARGET: Il fiume fino a Sonning serpeggia fra molte isole, ed è molto placido, raccolto e solitario.
    PREDICTED: Il fiume si avvicinò al fiume , e , dopo un poco , si , si , si e si .
--------------------------------------------------------------------------------
    SOURCE: 'I shall still get angry with Ivan the coachman in the same way, shall dispute in the same way, shall inopportunely express my thoughts; there will still be a wall between my soul's holy of holies and other people; even my wife I shall still blame for my own fears and shall repent of it. My reason will still not understand why I pray, but I shall still pray, and my life, my whole life, independently of anything that may happen to me, is every moment of it no longer meaningless as it was before, but has an unquestionable meaning of goodness with which I have the power to invest it.'
    TARGET: Mi arrabbierò sempre alla stessa maniera contro Ivan il cocchiere, sempre alla stessa maniera discuterò, esprimerò a sproposito le mie idee, ci sarà lo stesso muro fra il tempio dell’anima mia e quello degli altri, e perfino mia moglie accuserò sempre alla stessa maniera del mio spavento e ne proverò rimorso; sempre alla stessa maniera, non capirò con la ragione perché prego e intanto pregherò, ma la mia vita adesso, tutta la mia vita, indipendentemente da tutto quello che mi può accadere, ogni suo attimo, non solo non è più senza senso, come prima, ma ha un indubitabile senso di bene, che io ho il potere di trasfondere in essa!”.
    PREDICTED: — Io sono molto contento di andare in Russia , per il cocchiere , per il fratello , il mio desiderio , il mio desiderio , il mio desiderio , il mio desiderio , il mio desiderio , e per me ne sono un ’ altra , e per me ne , e non ci mai , ma non è possibile che il mio fratello non è possibile , ma che non è possibile , ma non è possibile , ma che non è possibile , ma non è possibile , ma non è possibile , ma che non è possibile .
--------------------------------------------------------------------------------
Processing Epoch 05: 100% 4850/4850 [04:52<00:00, 16.55it/s, loss=3.002]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: Knitting, sewing, reading, writing, ciphering, will be all you will have to teach.
    TARGET: Non dovete insegnar loro altro che a far la calza, a cucire, a leggere, a scrivere e a far di conto.
 PREDICTED: , , , tutti , tutto .
--------------------------------------------------------------------------------
    SOURCE: 'I will fetch it at once.
    TARGET: — La porto subito.
    PREDICTED: — Devo andare a casa .
--------------------------------------------------------------------------------
Processing Epoch 06: 100% 4850/4850 [04:45<00:00, 16.98it/s, loss=4.527]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: "And you see the candles?"
    TARGET: — E vedete anche le candele?
    PREDICTED: — E voi avete visto la testa ?
--------------------------------------------------------------------------------
    SOURCE: After a while, however, less grasping feelings prevailed.
    TARGET: Dopo un poco, però, prevalsero dei sentimenti meno esclusivi.
    PREDICTED: Dopo un certo modo , senza volere , si , e il suo corpo si .
--------------------------------------------------------------------------------
Processing Epoch 07: 100% 4850/4850 [04:50<00:00, 16.71it/s, loss=4.833]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: Each time he began to think about it, he felt that he must try again, that by kindness, tenderness, and persuasion there was still a hope of saving her and obliging her to bethink herself. Every day he prepared himself to have a talk with her.
    TARGET: Ogni qualvolta ci pensava, sentiva che era necessario tentare qualcosa, sentiva che, con la bontà, la tenerezza, la persuasione, c’era ancora la speranza di salvarla, di farla rientrare in sé, e ogni giorno si disponeva a parlare.
    PREDICTED: A ogni volta cominciò a pensare , cercando di capire che egli avrebbe provato , che la coscienza della disperazione , la coscienza della propria calma , e la sua calma , la tormentava , la voleva capire che cosa sarebbe stato un tempo di parlare .
--------------------------------------------------------------------------------
    SOURCE: 'I like it,' said Anna.
    TARGET: — No, mi piace — disse Anna.
    PREDICTED: — Io mi piace — disse Anna .
--------------------------------------------------------------------------------
Processing Epoch 08: 100% 4850/4850 [04:50<00:00, 16.72it/s, loss=4.062]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: 'But what has she done?' asked Levin rather indifferently. He wanted to consult her about his own affairs, and was annoyed at having come at an inopportune moment.
    TARGET: — Ma che ha fatto mai? — chiese Levin alquanto indifferente, e, desideroso di trovar consiglio per la cosa sua, s’irritò d’esser capitato fuor di proposito.
    PREDICTED: — Ma come mai ha fatto ? — disse Levin con stizza . — Per favore , è necessario condurre la sua azienda , e , dopo aver cercato di non dormire , si in un momento di modo .
--------------------------------------------------------------------------------
    SOURCE: She was not only glad to meet him, but searched his face for signs of the impression she created on him.
    TARGET: Ella non solo gioiva di un incontro con lui, ma cercava sul viso di lui i segni dell’impressione che lei stessa produceva.
    PREDICTED: Non solo ella si sentiva in lui , ma in quel momento , nel suo viso , si sentiva completamente liberato del proprio mutamento di lui .
--------------------------------------------------------------------------------
Processing Epoch 09: 100% 4850/4850 [04:52<00:00, 16.56it/s, loss=3.191]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: You might look daggers at him for an hour and he would not notice it, and it would not trouble him if he did.
    TARGET: Voi potevate lanciargli degli sguardi furiosi per un’ora, e lui non li vedeva, e non se ne sarebbe dato per inteso, se li avesse visti.
    PREDICTED: Se l ’ avesse guardato per un ’ ora , e lui non si sarebbe messo a guardarlo e se non lo avesse fatto .
--------------------------------------------------------------------------------
    SOURCE: And that punishment you made me suffer because your wicked boy struck me--knocked me down for nothing.
    TARGET: E quella punizione me l'avevate inflitta perché era stata percossa, gettata in terra dal vostro perfido figliuolo.
    PREDICTED: " E mi che il vostro ragazzo mi temete per nulla.
--------------------------------------------------------------------------------
Processing Epoch 10: 100% 4850/4850 [04:45<00:00, 16.99it/s, loss=3.816]
stty: 'standard input': Inappropriate ioctl for device
--------------------------------------------------------------------------------
    SOURCE: 'I don't think they play at all fairly,' Alice began, in rather a complaining tone, 'and they all quarrel so dreadfully one can't hear oneself speak--and they don't seem to have any rules in particular; at least, if there are, nobody attends to them--and you've no idea how confusing it is all the things being alive; for instance, there's the arch I've got to go through next walking about at the other end of the ground--and I should have croqueted the Queen's hedgehog just now, only it ran away when it saw mine coming!'
    TARGET: — Non credo che giochino realmente, — disse Alice lagnandosi. — Litigano con tanto calore che non sentono neanche la loro voce... non hanno regole nel giuoco; e se le hanno, nessuno le osserva... E poi c'è una tal confusione con tutti questi oggetti vivi; che non c'è modo di raccapezzarsi.
    PREDICTED: — Non credo che si a dire a Alice , — e poi si a dire che si possono parlare di questo . E poi , se non si possono parlare di quelle umane sono ; e siccome non ci sono mai più , e non ci sono più che per altro che una delle ragazze non ci : ecco , la Regina che si la Regina ; e siccome le acque , e le di là sul tavolo per terra .
--------------------------------------------------------------------------------
    SOURCE: 'My youngest,' replied the old man with a smile of affection.
    TARGET: — L’ultimo — disse il vecchio con un sorriso carezzevole.
    PREDICTED: — Il vecchio — disse , sorridendo con un sorriso gioioso .
--------------------------------------------------------------------------------
