The assignment was to use dataset from opus_books that too specific to english to french. The following steps were also required to complete the assignment:
1. Remove the English sentences with tokens more than 150.
2. Remove french sentences where len(fench_sentences) > len(english_sentrnce) + 10
3. One Cycle Policy. (Model trained for 30 epochs only).
Reducing the hidden layer in feed forward network from 1024 to 256.
Dynamic Padding.


Max length of source sentence token: 45
Max length of target sentence token: 48
Max length of source sentence: 149
Max length of target sentence: 158


Processing Epoch 00:  73% 4470/6150 [03:46<01:25, 19.64it/s, loss=4.747]IOPub message rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
To change this limit, set the config variable
`--NotebookApp.iopub_msg_rate_limit`.

Current values:
NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
NotebookApp.rate_limit_window=3.0 (secs)

Processing Epoch 01: 100% 6150/6150 [05:08<00:00, 19.94it/s, loss=4.031]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: "No," answered the reporter, "a few bruises only from the ricochet!
    TARGET: -- Non! répondit le reporter, quelques contusions seulement, par ricochet!
    PREDICTED: -- Non , répondit le reporter , un peu de quelques !
Count:  2
--------------------------------------------------------------------------------
    SOURCE: I had not been a week in my new office, when I happened to meet one evening a young Icoglan, extremely handsome and well-made.
    TARGET: Je fus nommé pour aller servir d'aumônier à Constantinople auprès de monsieur l'ambassadeur de France.
    PREDICTED: Je n ' avais pas été un jour , quand je l ' ai été déjà jeune , quand je restai jeune , très jeune , très jeune .
--------------------------------------------------------------------------------
Processing Epoch 02:  12% 741/6150 [00:38<04:33, 19.78it/s, loss=3.655]IOPub message rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
To change this limit, set the config variable
`--NotebookApp.iopub_msg_rate_limit`.

Current values:
NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
NotebookApp.rate_limit_window=3.0 (secs)

Processing Epoch 03:  49% 3001/6150 [02:30<02:34, 20.39it/s, loss=3.920]IOPub message rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
To change this limit, set the config variable
`--NotebookApp.iopub_msg_rate_limit`.

Current values:
NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
NotebookApp.rate_limit_window=3.0 (secs)

Processing Epoch 04:  79% 4853/6150 [04:06<01:05, 19.84it/s, loss=3.774]IOPub message rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
To change this limit, set the config variable
`--NotebookApp.iopub_msg_rate_limit`.

Current values:
NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
NotebookApp.rate_limit_window=3.0 (secs)

Processing Epoch 05: 100% 6150/6150 [05:13<00:00, 19.61it/s, loss=2.978]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: Having done all this I left them the next day, and went on board the ship.
    TARGET: Ceci fait, je pris congé d'eux le jour suivant, et m'en allai à bord du navire.
    PREDICTED: Tout cela fait , je les ai abandonné le lendemain , et je me rendis à bord du navire .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: Puis, levant les yeux sur notre héros, elle éclata de rire.
    TARGET: Then, raising her eyes to our hero, she burst out laughing.
    PREDICTED: Then she burst into eyes , she not of our hero .
--------------------------------------------------------------------------------
Processing Epoch 06:  15% 908/6150 [00:45<04:25, 19.77it/s, loss=2.923]IOPub message rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
To change this limit, set the config variable
`--NotebookApp.iopub_msg_rate_limit`.

Current values:
NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
NotebookApp.rate_limit_window=3.0 (secs)

Processing Epoch 07:  45% 2768/6150 [02:21<02:47, 20.15it/s, loss=2.568]IOPub message rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
To change this limit, set the config variable
`--NotebookApp.iopub_msg_rate_limit`.

Current values:
NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
NotebookApp.rate_limit_window=3.0 (secs)

Processing Epoch 08:  78% 4781/6150 [04:02<01:10, 19.31it/s, loss=2.617]IOPub message rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
To change this limit, set the config variable
`--NotebookApp.iopub_msg_rate_limit`.

Current values:
NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
NotebookApp.rate_limit_window=3.0 (secs)

Processing Epoch 09: 100% 6150/6150 [05:13<00:00, 19.62it/s, loss=2.519]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: You will find here, moreover, the young woman of whom I spoke, who is persecuted, no doubt, in consequence of some court intrigue.
    TARGET: Il y a plus, vous trouverez ici cette jeune femme persécutée sans doute par suite de quelque intrigue de cour.
    PREDICTED: Vous trouverez ici , d ’ ailleurs la jeune femme que je parlais , qui est sans doute , il y a quelque doute dans quelque sorte de la cour .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: 'This is the end of everything,' cried Madame de Renal, throwing herself into Julien's arms.
    TARGET: – Voici la fin de tout, s’écria Mme de Rênal, en se jetant dans les bras de Julien.
    PREDICTED: Voilà tout ! s ’ écria Mme de Rênal en se jetant dans les bras de Julien .
--------------------------------------------------------------------------------
Processing Epoch 10:   9% 563/6150 [00:28<04:43, 19.68it/s, loss=2.076]IOPub message rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
To change this limit, set the config variable
`--NotebookApp.iopub_msg_rate_limit`.

Current values:
NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
NotebookApp.rate_limit_window=3.0 (secs)

Processing Epoch 11:  39% 2400/6150 [02:05<03:16, 19.08it/s, loss=1.878]IOPub message rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
To change this limit, set the config variable
`--NotebookApp.iopub_msg_rate_limit`.

Current values:
NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
NotebookApp.rate_limit_window=3.0 (secs)

Processing Epoch 11: 100% 6150/6150 [05:18<00:00, 19.32it/s, loss=2.340]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: "Exquisite!" Conseil replied.
    TARGET: -- Exquis ! répondait Conseil.
    PREDICTED: -- ! répondit Conseil .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: But in matters of greater weight, I may suffer from want of money.
    TARGET: Mais le manque de fortune peut m’exposer a des épreuves plus graves.
    PREDICTED: Mais dans les intérêts de la , je ne puis souffrir du reste .
--------------------------------------------------------------------------------
Processing Epoch 12: 100% 6150/6150 [05:15<00:00, 19.46it/s, loss=1.676]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: He pressed a metal button and at once the propeller slowed down significantly.
    TARGET: Il pressa un bouton de métal, et aussitôt la vitesse de l'hélice fut très diminuée.
    PREDICTED: Il pressa un bouton de métal , et , il remit tout de suite , il his hélice .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: What were you doing at that window?"
    TARGET: Que faisiez-vous à cette fenêtre ? »
    PREDICTED: Que faites - vous donc à cette lucarne ?
--------------------------------------------------------------------------------
Processing Epoch 13: 100% 6150/6150 [05:13<00:00, 19.62it/s, loss=1.980]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: It was the end of November and Holmes and I sat, upon a raw and foggy night, on either side of a blazing fire in our sitting-room in Baker Street.
    TARGET: Fin novembre, Holmes et moi étions assis de chaque côté d’un bon feu dans notre petit salon de Baker Street ; dehors la nuit était rude, brumeuse.
    PREDICTED: C ’ était la fin du novembre et Holmes , assis sur une nuit noire , et nuit de côté , à nos pas de Baker Street , nos retour .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: Charles was silent.
    TARGET: Charles se taisait.
    PREDICTED: Charles se tut .
--------------------------------------------------------------------------------
Processing Epoch 14: 100% 6150/6150 [05:13<00:00, 19.61it/s, loss=1.784]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: Go on!'
    TARGET: Continuez ! »
    PREDICTED: Va ! »
Count:  2
--------------------------------------------------------------------------------
    SOURCE: Catherine bent forward and said in Étienne's ear:
    TARGET: Catherine se pencha, dit a l'oreille d'Étienne:
    PREDICTED: Catherine s ' avança et dit a l ' oreille d ' Étienne .
--------------------------------------------------------------------------------
Processing Epoch 15: 100% 6150/6150 [05:16<00:00, 19.42it/s, loss=1.791]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: But I have not a minute to-day.
    TARGET: Mais je n'ai pas le tempsaujourd'hui.
    PREDICTED: Mais je n ’ ai pas une minute aujourd ’ hui .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: "One is the scullery-maid, who sleeps in the other wing.
    TARGET: L’une est la laveuse de vaisselle, qui couche dans l’autre aile.
    PREDICTED: -- On est la fille de chambre de fille , qui dort à l ' autre .
--------------------------------------------------------------------------------
Processing Epoch 16: 100% 6150/6150 [05:14<00:00, 19.56it/s, loss=1.959]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: Tomorrow, on the day stated and at the hour stated, the tide will peacefully lift it off, and it will resume its navigating through the seas."
    TARGET: Demain, au jour dit, à l'heure dite, la marée le soulèvera paisiblement, et il reprendra sa navigation à travers les mers.
    PREDICTED: Demain , le jour , et , devant la heure , la marée paisiblement , s ' , et elle sa navigation à travers les mers .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: "You will visit each of these in turn."
    TARGET: – Vous les visiterez à tour de rôle.
    PREDICTED: -- Vous à propos de ce tour .
--------------------------------------------------------------------------------
Processing Epoch 17: 100% 6150/6150 [05:16<00:00, 19.41it/s, loss=1.723]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: Let's forge ahead!
    TARGET: Allons en avant !
    PREDICTED: Continuons la forge !
Count:  2
--------------------------------------------------------------------------------
    SOURCE: "And so have I, sir," I returned, putting my hands and my purse behind me. "I could not spare the money on any account."
    TARGET: -- Et moi aussi, monsieur, répondis-je en cachant ma bourse, je ne pourrais pas un instant me passer de cet argent.
    PREDICTED: -- Et moi aussi , monsieur , répliquai - je , en portant ma bourse derriere moi .
--------------------------------------------------------------------------------
Processing Epoch 18: 100% 6150/6150 [05:16<00:00, 19.41it/s, loss=1.928]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: A sort of uneasiness had seized Pencroft upon the subject of his vessel.
    TARGET: Une sorte d'inquiétude avait pris Pencroff au sujet de son embarcation.
    PREDICTED: Une sorte de inquiétude avait donc à Pencroff de son compagnon .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: Julien, without exactly knowing what he was doing, grasped her hand again.
    TARGET: Julien, sans trop savoir ce qu’il faisait, la saisit de nouveau.
    PREDICTED: Julien , sans trop savoir ce qu ’ il faisait , lui prit la main des mains .
--------------------------------------------------------------------------------
Processing Epoch 19: 100% 6150/6150 [05:18<00:00, 19.33it/s, loss=1.842]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: — Bon, dit-il.
    TARGET: "That's good enough.
    PREDICTED: " Good ," he said .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: "Doubtless," said Buckingham, "and rather twice than once."
    TARGET: -- Sans doute, dit Buckingham, et plutôt deux fois qu'une.
    PREDICTED: -- Sans doute , dit Buckingham , et un peu d ' ensemble .
--------------------------------------------------------------------------------
Processing Epoch 20: 100% 6150/6150 [05:15<00:00, 19.50it/s, loss=1.697]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: Julien's attention was sharply distracted by the almost immediate arrival of a wholly different person.
    TARGET: L’attention de Julien fut vivement distraite par l’arrivée presque immédiate d’un être tout différent.
    PREDICTED: L ’ attention de Julien fut très vivement du arrivée à l ’ arrivée d ’ une personne très différente .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: "A good voyage to you," shouted the sailor, who himself did not expect any great result from this mode of correspondence.
    TARGET: -- Bon voyage!» s'écria le marin, qui, lui, n'attendait pas grand résultat de ce mode de correspondance.
    PREDICTED: -- Une bonne traversée , fit le marin , qui ne s ' attendait guère à ce lieu de la haute en cet ordre .
--------------------------------------------------------------------------------
Processing Epoch 21: 100% 6150/6150 [05:16<00:00, 19.44it/s, loss=1.771]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: Thousands of luminous sheets and barbed tongues of fire were cast in various directions.
    TARGET: Des milliers de fragments lumineux et de points vifs se projetaient en directions contraires.
    PREDICTED: Un mille facettes de Sire lumineux et de morts furent à quelques instructions .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: In London he at last made acquaintance with the extremes of fatuity.
    TARGET: À Londres, il connut enfin la haute fatuité.
    PREDICTED: Enfin il fit connaissance avec la connaissance des , en finissait par l ’ enfant .
--------------------------------------------------------------------------------
Processing Epoch 22: 100% 6150/6150 [05:15<00:00, 19.50it/s, loss=1.643]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: Did you ever know one in your life that was transported and had a hundred pounds in his pocket, I'll warrant you, child?' says she.
    TARGET: --Mais tu as de l'argent, n'est-ce pas? En as-tu déjà connu une dans ta vie qui se fît déporter avec 100£ dans sa poche?
    PREDICTED: - vous jamais dans votre vie de ce travail des et bien à cent livres ( je te l ’ dirai , dites - vous ?
Count:  2
--------------------------------------------------------------------------------
    SOURCE: "Come, that I may tell you that very softly.
    TARGET: « Venez, que je vous dise cela tout bas.
    PREDICTED: – Allons , que je puis vous dire cela très bas .
--------------------------------------------------------------------------------
Processing Epoch 23: 100% 6150/6150 [05:15<00:00, 19.50it/s, loss=1.854]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: He was afraid of bringing everything to an end by a sudden concession.
    TARGET: Il avait peur de tout finir par une concession subite.
    PREDICTED: Il craignait de faire tout porter à une extrémité inférieure .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: They were now outside the forest, at the beginning of the powerful spurs which supported Mount Franklin towards the west.
    TARGET: Il se trouvait en dehors de la forêt, à la naissance de ces puissants contreforts qui étançonnaient le mont Franklin vers l'est.
    PREDICTED: Ils étaient donc en dehors la forêt , au commencement des contreforts , que le mont Franklin donnait sur l ' ouest .
--------------------------------------------------------------------------------
Processing Epoch 24: 100% 6150/6150 [05:13<00:00, 19.61it/s, loss=1.699]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: "Their conduct has been such," replied Elizabeth, "as neither you, nor I, nor anybody can ever forget.
    TARGET: – Leur conduite a été telle, répliqua Elizabeth, que ni vous, ni moi, ni personne ne pourrons jamais l’oublier.
    PREDICTED: – Cette conduite a été d ’ autant , répondit Elizabeth , et je n ’ ai jamais rien d ’ oublier .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: There are lonely houses scattered over the moor, and he is a fellow who would stick at nothing.
    TARGET: Il y a des maisons isolées sur la lande, et il ferait n’importe quoi.
    PREDICTED: Il y a des maisons qui sortent , et c ’ est un homme qui ne s ’ rien .
--------------------------------------------------------------------------------
Processing Epoch 25: 100% 6150/6150 [05:14<00:00, 19.58it/s, loss=1.991]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: The old man shrugged his shoulders, and then let them fall as if overwhelmed beneath an avalanche of gold.
    TARGET: Le vieux haussa les épaules, puis les laissa retomber, comme accablé sous un écroulement d'écus.
    PREDICTED: Le vieux eut un haussement d ' épaules , puis qu ' on se comme sous un de l ' or .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: There were reasons why I could not get there earlier."
    TARGET: Voilà pourquoi je ne pouvais pas me rendre plus tôt au manoir.
    PREDICTED: J ’ ai des raisons que je ne pouvais voir là - bas .
--------------------------------------------------------------------------------
Processing Epoch 26: 100% 6150/6150 [05:13<00:00, 19.61it/s, loss=1.771]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: "I pardon you, monseigneur!" said Bonacieux, hesitating to take the purse, fearing, doubtless, that this pretended gift was but a pleasantry.
    TARGET: -- Que je vous pardonne, Monseigneur! dit Bonacieux hésitant à prendre le sac, craignant sans doute que ce prétendu don ne fût qu'une plaisanterie.
    PREDICTED: -- Je vous le pardonne , Monseigneur , dit Bonacieux en sûre , sans crainte , que ce don ' t be une plaisanterie .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: "Jane, are you ready?"
    TARGET: -- Jane, êtes-vous prête?
    PREDICTED: « Jane , êtes - vous prêt ?
--------------------------------------------------------------------------------
Processing Epoch 27: 100% 6150/6150 [05:17<00:00, 19.37it/s, loss=1.574]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: "A vessel from the Vineyard!
    TARGET: «Un navire du Vineyard!
    PREDICTED: -- Un navire du Vineyard ?
Count:  2
--------------------------------------------------------------------------------
    SOURCE: She repeated, "He is out."
    TARGET: Elle répéta: -- Il est absent.
    PREDICTED: Elle répéta : -- Il est absent .
--------------------------------------------------------------------------------
Processing Epoch 28: 100% 6150/6150 [05:18<00:00, 19.34it/s, loss=1.654]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: It was a bay horse hardly three years of age, called Trompette.
    TARGET: C'était un cheval bai, de trois ans a peine, nommé Trompette.
    PREDICTED: C ' était un cheval de cinq ans , on ne l ' appelle Trompette .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: "Yes, but yesterday at five o’clock in the afternoon, thanks to you, she escaped."
    TARGET: -- Oui, mais depuis hier cinq heures de l'après-midi, grâce à vous, elle s'est échappée.
    PREDICTED: -- Oui , mais hier à cinq heures du soir , merci , elle vous a sauvé .
--------------------------------------------------------------------------------
Processing Epoch 29: 100% 6150/6150 [05:20<00:00, 19.18it/s, loss=1.715]
stty: 'standard input': Inappropriate ioctl for device
Count:  1
--------------------------------------------------------------------------------
    SOURCE: Elizabeth disdained the appearance of noticing this civil reflection, but its meaning did not escape, nor was it likely to conciliate her.
    TARGET: Elizabeth parut dédaigner cette réflexion aimable mais le sens ne lui en échappa point et, de plus en plus animée, elle reprit :
    PREDICTED: Elizabeth , qui s ’ en aperçut de cette réflexion n ’ avait pas dit , le ou l ’ on veut se passer de ce sur la premiere Darcy .
Count:  2
--------------------------------------------------------------------------------
    SOURCE: I remember my brother-in-law going for a short sea trip once, for the benefit of his health.
    TARGET: Un jour, mon beau-frere fit une petite croisiere en mer, pour sa santé.
 PREDICTED: Je me rappelle mon beau - frère , avoir la mer pour le , car la santé de sa santé .
