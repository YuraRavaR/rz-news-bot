# Rz-Flow

Rz-Flow reads local news sites from Rzeszów, lets an AI decide which articles are worth showing, and posts the chosen ones in Ukrainian to Telegram channels.

## News

**Source** (Джерело):
A news site the bot reads.
_Avoid_: portal, feed, site

**Article** (Стаття):
One news page on one Source. The same real-world news on two Sources is two Articles.
_Avoid_: news, item, post

**Story** (Сюжет):
The real-world news or event that one or more Articles report on, within a few days. A follow-up that brings new information (a festival opening, then its wrap-up) is a separate Story, and so is a recurring happening with the same name (another water outage).
_Avoid_: news, topic

**Duplicate** (Дубль):
Two Posts in the same Channel about the same Story, whether they come from one Article or from several.

## Channels

**Main channel** (Головний канал):
The public channel every Published article goes to.

**Events channel** (Канал подій):
A secondary channel that additionally gets Articles about upcoming events. Reaching it alone does not make an Article published.

**Event date** (Дата події):
The day an event takes place, or its first and last day when it lasts several days. The AI reads it from the Article's title and lead, resolving relative dates ("w sobotę") against the day of the Run. When the text gives no date, the event has none: a date is never guessed. Events channel Posts show it; Main channel Posts do not.
_Avoid_: deadline, publication date

**Digest** (Дайджест):
One Events channel Post listing the coming weekend's events (Friday to Sunday, Europe/Warsaw), grouped by day, each linking to its Events channel Post (or to its Article when it has none). It lists Published events whose Event date overlaps the weekend (events without one are left out) and the weekend's Calendar entries, translated into Ukrainian, that the AI finds interesting and not the same event as a listed news event; those link to their Calendar page. It is posted at most once per weekend, and not at all with fewer than three items.
_Avoid_: summary, newsletter, roundup

**Calendar** (Календар подій):
The city hall's events calendar on erzeszow.pl, listing events of city institutions (libraries, culture houses, galleries). It is read only for the Digest, one request per weekend day, and is not a Source: it never goes through a Run, the Repeat check or storage.
_Avoid_: source, feed, schedule

**Calendar entry** (Запис календаря):
One event listed in the Calendar, with a Polish title, a start day, an optional last day and a link. It is not an Article. The Digest considers an entry that lasts at most three days and overlaps the weekend, or a longer one only in the weekend it starts, so a months-long exhibition appears once.
_Avoid_: article, event (on its own)

**Weekly report** (Тижневий звіт):
A report to the admin chat about the previous Monday–Sunday (Europe/Warsaw), built from storage alone: per Source, how many Articles were new, Published, Skipped, Repeats, Errors and Unconfirmed; events posted and how many had no Event date; an estimate of Gemini requests on the busiest day against the free-tier limit; whether that weekend's Digest went out; and a warning when Published fell by more than half against the week before. An Article counts in the week of its last write. It costs no Gemini requests.
_Avoid_: digest (that is the Events channel Post), run report (that covers one Run)

**Post** (Пост):
One message in one Channel.
_Avoid_: publication, message

**Published article** (Опублікована стаття):
An Article that has a Post in the Main channel.

## Outcomes

**Skipped** (Пропущена):
The AI evaluated the Article and turned it down. The decision is final.
_Avoid_: rejected, filtered

**Repeat** (Повтор):
An Article not sent to the AI because the AI has already judged another Article about the same Story. It is final and always shown to the admin next to its original.
_Avoid_: duplicate (that word means two Posts)

**Deferred** (Відкладена):
The Article could not be processed because of a temporary problem that is not its fault (AI or Telegram limits, an AI outage, a cut-off AI answer), and no Post went out. Storage stays as it was before the Run, and the next Run picks the Article up again without spending its attempts.
_Avoid_: skipped, postponed

**Error** (Помилка):
Processing failed and the cause may lie with the Article. It is retried on later Runs until its attempts run out.
_Avoid_: failed

**Publishing** (Публікується):
An Article whose Post was attempted but not confirmed. It is never posted again automatically, because the Post may already be in the Channel.
_Avoid_: pending, in progress

**Unconfirmed** (Непідтверджена):
A Publishing Article that a later Run found unresolved. It is final, and the admin is told once so they can check the Channel.
_Avoid_: lost, orphaned

**Settled** (Завершена):
An Article that will not be processed again: Published, Skipped, Repeat, Unconfirmed, or an Error that ran out of attempts.

**Run** (Запуск):
One scheduled or manual pass over all Sources.
