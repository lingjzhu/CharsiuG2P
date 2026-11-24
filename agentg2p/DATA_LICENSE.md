# Data sources and licences

The MIT licence in [LICENSE](LICENSE) covers the **code** only. The data on the GitHub
releases (dictionaries and the models trained on them) comes from these sources and keeps
their terms:

| Source | Dictionaries | Where it comes from |
| :--- | ---: | :--- |
| Montreal Forced Aligner (MFA) pronunciation dictionaries | 73 | <https://mfa-models.readthedocs.io/>, <https://github.com/MontrealCorpusTools/mfa-models> |
| WikiPron | 546 | <https://github.com/CUNY-CL/wikipron>, mined from Wiktionary (<https://www.wiktionary.org/>) |
| CharsiuG2P | 98 | <https://github.com/lingjzhu/CharsiuG2P>, which collects dictionaries from several sources |

- **WikiPron:** Wiktionary content is licensed under the Creative Commons
  Attribution-ShareAlike licence (CC BY-SA). Data derived from it, including the WikiPron
  lexicons here and models trained only on them, should be shared under the same terms,
  with attribution to Wiktionary's contributors.
- **MFA and CharsiuG2P:** see each project for its terms and for the original sources of
  individual dictionaries.
- **Models:** the FST models are trained per dictionary. The ByT5 models are trained on the
  MFA, WikiPron and Charsiu dictionaries together ("pooled") or on one source at a time.

The project's maintainers confirmed on 2026-11-19 that these licences permit redistribution.
If you are a rights holder and see a problem, please open an issue.

Please cite the original resources when you use this data:

- Lee, J. L., Ashby, L. F. E., Garza, M. E., Lee-Sikka, Y., Miller, S., Wong, A.,
  McCarthy, A. D., and Gorman, K. (2020). Massively multilingual pronunciation modeling
  with WikiPron. *LREC 2020*.
- McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., and Sonderegger, M. (2017). Montreal
  Forced Aligner. *Interspeech 2017*.
- Zhu, J., Zhang, C., and Jurgens, D. (2022). ByT5 model for massively multilingual
  grapheme-to-phoneme conversion. *Interspeech 2022*.
- Novak, J. R., Minematsu, N., and Hirose, K. (2016). Phonetisaurus: Exploring
  grapheme-to-phoneme conversion with joint n-gram models in the WFST framework.
  *Natural Language Engineering*, 22(6).
