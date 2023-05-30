CDF  �   
      time       bnds         6   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-ESM2-1 (2017):  aerosol: TACTIC_v2 atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: REPROBUS-C_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) ocnBgchem: Pisces 2.s seaIce: Gelato 6.1    institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    creation_date         2018-04-23T09:02:27Z   description       DECK: control      title         <CNRM-ESM2-1 model output prepared for CMIP6 / CMIP piControl   activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.1    experiment_id         	piControl      
experiment        pre-industrial control     forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-ESM2-1.piControl.none.r1i1p1f2      grid      2native ocean tri-polar grid with 105 k ocean cells     
grid_label        gn     nominal_resolution        100 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      CNRM-ESM2-1    parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent         @�Հ       branch_time_in_child                 physics_index               product       model-output   realization_index               realm         ocean      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         CNRM-ESM2-1    source_type       AOGCM BGC AER CHEM     sub_experiment_id         none   sub_experiment        none   table_id      Omon   variable_id       zostoga    variant_info      �. Information provided by this attribute may in some cases be flawed. Users can find more comprehensive and up-to-date documentation via the further_info_url global attribute.    variant_label         r1i1p1f2   EXPID         CNRM-ESM2-1_piControl_r1i1p1f2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          87374385b726e2a5f1e17b33af88ce8c   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.1      history      �Tue May 30 16:59:23 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.zostoga.gn.v20181115/zostoga_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.1d.yearmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_mergetime/zostoga/CNRM-ESM2-1_r1i1p1f2/zostoga_CNRM-ESM2-1_r1i1p1f2_piControl.mergetime.nc
Thu Apr 07 22:51:45 2022: cdo -O -s --reduce_dim -selname,zostoga -yearmean /Users/benjamin/Data/p22b/CMIP6/zostoga/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.zostoga.gn.v20181115/zostoga_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.zostoga.gn.v20181115/zostoga_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.1d.yearmean.nc
none    tracking_id       1hdl:21.14100/d3dca963-5f67-489f-8781-f8e4b46e6c4b      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               0   	time_bnds                                 8   zostoga                 standard_name         ,global_average_thermosteric_sea_level_change   	long_name         ,Global Average Thermosteric Sea Level Change   units         m      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    online_operation      average    interval_operation        1800 s     interval_write        1 month    description       /There is no CMIP6 request for zosga nor zossga.    history       none            H   none            HAq���   Aq��P   Aq�P   ����Aq�6�   Aq�P   Aq��P   ���Aq���   Aq��P   Aq��P   ����Aq��   Aq��P   Aq�dP   ����Aq���   Aq�dP   Aq��P   ����Aq���   Aq��P   Aq�FP   ����Aq�k�   Aq�FP   Aq��P   ���Aq���   Aq��P   Aq�(P   ���JAq�M�   Aq�(P   Aq��P   ����Aq���   Aq��P   Aq�
P   ����Aq�/�   Aq�
P   Aq�{P   ���UAq���   Aq�{P   Aq��P   ����Aq��   Aq��P   Aq�]P   ���GAqĂ�   Aq�]P   Aq��P   ���Aq���   Aq��P   Aq�?P   ���
Aq�d�   Aq�?P   Aq˰P   ����Aq���   Aq˰P   Aq�!P   ����Aq�F�   Aq�!P   AqВP   ���Aqз�   AqВP   Aq�P   ���KAq�(�   Aq�P   Aq�tP   ���6Aqՙ�   Aq�tP   Aq��P   ���>Aq�
�   Aq��P   Aq�VP   ����Aq�{�   Aq�VP   Aq��P   ����Aq���   Aq��P   Aq�8P   ����Aq�]�   Aq�8P   Aq�P   ����Aq���   Aq�P   Aq�P   ����Aq�?�   Aq�P   Aq�P   ����Aq��   Aq�P   Aq��P   ���]Aq�!�   Aq��P   Aq�mP   ��.Aq��   Aq�mP   Aq��P   ���iAq��   Aq��P   Aq�OP   ���Aq�t�   Aq�OP   Aq��P   ��CAq���   Aq��P   Aq�1P   ��TAq�V�   Aq�1P   Aq��P   ��{ Aq���   Aq��P   Aq�P   ��tmAq�8�   Aq�P   Aq��P   ��t�Aq���   Aq��P   Aq��P   ��oAq��   Aq��P   ArfP   ��l�Ar��   ArfP   Ar�P   ��m�Ar��   Ar�P   ArHP   ��n�Arm�   ArHP   Ar�P   ��qIAr��   Ar�P   Ar*P   ��s�ArO�   Ar*P   Ar�P   ��odAr��   Ar�P   ArP   ��t�Ar1�   ArP   Ar}P   ��v�Ar��   Ar}P   Ar�P   ��}RAr�   Ar�P   Ar_P   ��z�Ar��   Ar_P   Ar�P   ��}�Ar��   Ar�P   ArAP   ���=Arf�   ArAP   Ar�P   ���MAr��   Ar�P   Ar!#P   ����Ar!H�   Ar!#P   Ar#�P   ���tAr#��   Ar#�P   Ar&P   ���MAr&*�   Ar&P   Ar(vP   ���BAr(��   Ar(vP   Ar*�P   ����Ar+�   Ar*�P   Ar-XP   ����Ar-}�   Ar-XP   Ar/�P   ���$Ar/��   Ar/�P   Ar2:P   ���XAr2_�   Ar2:P   Ar4�P   ��Ar4��   Ar4�P   Ar7P   ��w?Ar7A�   Ar7P   Ar9�P   ��m:Ar9��   Ar9�P   Ar;�P   ��x�Ar<#�   Ar;�P   Ar>oP   ��vAr>��   Ar>oP   Ar@�P   ��r�ArA�   Ar@�P   ArCQP   ��l�ArCv�   ArCQP   ArE�P   ��o�ArE��   ArE�P   ArH3P   ��qtArHX�   ArH3P   ArJ�P   ��j!ArJ��   ArJ�P   ArMP   ��lmArM:�   ArMP   ArO�P   ��i{ArO��   ArO�P   ArQ�P   ��c�ArR�   ArQ�P   ArThP   ��j�ArT��   ArThP   ArV�P   ��i�ArV��   ArV�P   ArYJP   ��`EArYo�   ArYJP   Ar[�P   ��^�Ar[��   Ar[�P   Ar^,P   ��_�Ar^Q�   Ar^,P   Ar`�P   ��f�Ar`��   Ar`�P   ArcP   ��i�Arc3�   ArcP   AreP   ��h�Are��   AreP   Arg�P   ��h�Arh�   Arg�P   ArjaP   ��b�Arj��   ArjaP   Arl�P   ��\Arl��   Arl�P   AroCP   ��]�Aroh�   AroCP   Arq�P   ��_�Arq��   Arq�P   Art%P   ��]�ArtJ�   Art%P   Arv�P   ��XqArv��   Arv�P   AryP   ��Z�Ary,�   AryP   Ar{xP   ��W�Ar{��   Ar{xP   Ar}�P   ��S�Ar~�   Ar}�P   Ar�ZP   ��NAr��   Ar�ZP   Ar��P   ��X	Ar���   Ar��P   Ar�<P   ��Y�Ar�a�   Ar�<P   Ar��P   ��L�Ar���   Ar��P   Ar�P   ��M�Ar�C�   Ar�P   Ar��P   ��L�Ar���   Ar��P   Ar� P   ��H�Ar�%�   Ar� P   Ar�qP   ��D�Ar���   Ar�qP   Ar��P   ��MAr��   Ar��P   Ar�SP   ��L�Ar�x�   Ar�SP   Ar��P   ��G�Ar���   Ar��P   Ar�5P   ��GVAr�Z�   Ar�5P   Ar��P   ��K�Ar���   Ar��P   Ar�P   ��I�Ar�<�   Ar�P   Ar��P   ��HFAr���   Ar��P   Ar��P   ��I7Ar��   Ar��P   Ar�jP   ��GeAr���   Ar�jP   Ar��P   ��EsAr� �   Ar��P   Ar�LP   ��H�Ar�q�   Ar�LP   Ar��P   ��L�Ar���   Ar��P   Ar�.P   ��H*Ar�S�   Ar�.P   Ar��P   ��INAr���   Ar��P   Ar�P   ��H�Ar�5�   Ar�P   Ar��P   ��GAr���   Ar��P   Ar��P   ��A�Ar��   Ar��P   Ar�cP   ��BAr���   Ar�cP   Ar��P   ��=aAr���   Ar��P   Ar�EP   ��8iAr�j�   Ar�EP   ArĶP   ��2|Ar���   ArĶP   Ar�'P   ��:Ar�L�   Ar�'P   ArɘP   ��:Arɽ�   ArɘP   Ar�	P   ��4�Ar�.�   Ar�	P   Ar�zP   ��4�ArΟ�   Ar�zP   Ar��P   ��9�Ar��   Ar��P   Ar�\P   ��@EArӁ�   Ar�\P   Ar��P   ��9:Ar���   Ar��P   Ar�>P   ��0(Ar�c�   Ar�>P   ArگP   ��7*Ar���   ArگP   Ar� P   ��6�Ar�E�   Ar� P   ArߑP   ��3�Ar߶�   ArߑP   Ar�P   ��.LAr�'�   Ar�P   Ar�sP   ��/�Ar��   Ar�sP   Ar��P   ��/�Ar�	�   Ar��P   Ar�UP   ��8cAr�z�   Ar�UP   Ar��P   ��4Ar���   Ar��P   Ar�7P   ��([Ar�\�   Ar�7P   Ar�P   ��#hAr���   Ar�P   Ar�P   ��+Ar�>�   Ar�P   Ar��P   ��+�Ar���   Ar��P   Ar��P   ��#�Ar� �   Ar��P   Ar�lP   ��#�Ar���   Ar�lP   Ar��P   ��%/Ar��   Ar��P   Ar�NP   ��(XAr�s�   Ar�NP   As�P   ��$EAs��   As�P   As0P   �� 2AsU�   As0P   As�P   �� As��   As�P   As	P   ���As	7�   As	P   As�P   ��pAs��   As�P   As�P   ���As�   As�P   AseP   ���As��   AseP   As�P   ��SAs��   As�P   AsGP   ���Asl�   AsGP   As�P   ���As��   As�P   As)P   �� �AsN�   As)P   As�P   ��As��   As�P   AsP   ��
�As0�   AsP   As!|P   ��vAs!��   As!|P   As#�P   ��As$�   As#�P   As&^P   ��
pAs&��   As&^P   As(�P   ��)As(��   As(�P   As+@P   ��GAs+e�   As+@P   As-�P   ��As-��   As-�P   As0"P   ��mAs0G�   As0"P   As2�P   ��%�As2��   As2�P   As5P   ��&�As5)�   As5P   As7uP   ���As7��   As7uP   As9�P   ��!�As:�   As9�P   As<WP   ��$As<|�   As<WP   As>�P   �� �As>��   As>�P   AsA9P   ��!~AsA^�   AsA9P   AsC�P   ���AsC��   AsC�P   AsFP   �� hAsF@�   AsFP   AsH�P   ��!�AsH��   AsH�P   AsJ�P   ��"AsK"�   AsJ�P   AsMnP   ��WAsM��   AsMnP   AsO�P   ��HAsP�   AsO�P   AsRPP   ��8AsRu�   AsRPP   AsT�P   ��AAsT��   AsT�P   AsW2P   ��TAsWW�   AsW2P   AsY�P   ��YAsY��   AsY�P   As\P   ��bAs\9�   As\P   As^�P   ��lAs^��   As^�P   As`�P   ���Asa�   As`�P   AscgP   ���Asc��   AscgP   Ase�P   ���Ase��   Ase�P   AshIP   ��}Ashn�   AshIP   Asj�P   ���Asj��   Asj�P   Asm+P   ��
�AsmP�   Asm+P   Aso�P   ���Aso��   Aso�P   AsrP   ��	�Asr2�   AsrP   Ast~P   ��_Ast��   Ast~P   Asv�P   ��%Asw�   Asv�P   Asy`P   ��6Asy��   Asy`P   As{�P   ��As{��   As{�P   As~BP   ���As~g�   As~BP   As��P   ���As���   As��P   As�$P   ��+As�I�   As�$P   As��P   ��5As���   As��P   As�P   ����As�+�   As�P   As�wP   ��GAs���   As�wP   As��P   ��	As��   As��P   As�YP   ���As�~�   As�YP   As��P   �� �As���   As��P   As�;P   ����As�`�   As�;P   As��P   ���)As���   As��P   As�P   ���As�B�   As�P   As��P   ���jAs���   As��P   As��P   ���As�$�   As��P   As�pP   ���UAs���   As�pP   As��P   ����As��   As��P   As�RP   ����As�w�   As�RP   As��P   ���rAs���   As��P   As�4P   ���AAs�Y�   As�4P   As��P   ���]As���   As��P   As�P   ���aAs�;�   As�P   As��P   ����As���   As��P   As��P   ����As��   As��P   As�iP   ���As���   As�iP   As��P   ���wAs���   As��P   As�KP   ����As�p�   As�KP   As��P   ���As���   As��P   As�-P   ���mAs�R�   As�-P   AsP   ���As���   AsP   As�P   ���As�4�   As�P   AsǀP   ���QAsǥ�   AsǀP   As��P   ���As��   As��P   As�bP   ����Aṡ�   As�bP   As��P   ����As���   As��P   As�DP   ���ZAs�i�   As�DP   AsӵP   ���As���   AsӵP   As�&P   ���
As�K�   As�&P   AsؗP   ����Asؼ�   AsؗP   As�P   ���As�-�   As�P   As�yP   ���9Asݞ�   As�yP   As��P   ���MAs��   As��P   As�[P   ���BAs��   As�[P   As��P   ���2As���   As��P   As�=P   ���+As�b�   As�=P   As�P   ���As���   As�P   As�P   ���As�D�   As�P   As�P   ���As��   As�P   As�P   ����As�&�   As�P   As�rP   ���As��   As�rP   As��P   ���As��   As��P   As�TP   ���PAs�y�   As�TP   As��P   ���7As���   As��P   As�6P   ���hAs�[�   As�6P   As��P   ���zAs���   As��P   AtP   ����At=�   AtP   At�P   ���!At��   At�P   At�P   ����At�   At�P   At	kP   ���ZAt	��   At	kP   At�P   ����At�   At�P   AtMP   ��נAtr�   AtMP   At�P   ��ہAt��   At�P   At/P   ���?AtT�   At/P   At�P   ����At��   At�P   AtP   ����At6�   AtP   At�P   ���At��   At�P   At�P   ���pAt�   At�P   AtdP   ����At��   AtdP   At!�P   ���XAt!��   At!�P   At$FP   ���UAt$k�   At$FP   At&�P   ��®At&��   At&�P   At)(P   ����At)M�   At)(P   At+�P   ����At+��   At+�P   At.
P   ���At./�   At.
P   At0{P   ���{At0��   At0{P   At2�P   ���0At3�   At2�P   At5]P   ����At5��   At5]P   At7�P   ���At7��   At7�P   At:?P   ���}At:d�   At:?P   At<�P   ���At<��   At<�P   At?!P   ����At?F�   At?!P   AtA�P   ����AtA��   AtA�P   AtDP   ���_AtD(�   AtDP   AtFtP   ���EAtF��   AtFtP   AtH�P   ���AtI
�   AtH�P   AtKVP   ���4AtK{�   AtKVP   AtM�P   ����AtM��   AtM�P   AtP8P   ����AtP]�   AtP8P   AtR�P   ���-AtR��   AtR�P   AtUP   ���AtU?�   AtUP   AtW�P   ���XAtW��   AtW�P   AtY�P   ���wAtZ!�   AtY�P   At\mP   ����At\��   At\mP   At^�P   ���At_�   At^�P   AtaOP   ���ZAtat�   AtaOP   Atc�P   ���Atc��   Atc�P   Atf1P   ����AtfV�   Atf1P   Ath�P   ����Ath��   Ath�P   AtkP   ���_Atk8�   AtkP   Atm�P   ���aAtm��   Atm�P   Ato�P   ���SAtp�   Ato�P   AtrfP   ���cAtr��   AtrfP   Att�P   ��}Att��   Att�P   AtwHP   ��z�Atwm�   AtwHP   Aty�P   ��w�Aty��   Aty�P   At|*P   ��tRAt|O�   At|*P   At~�P   ��skAt~��   At~�P   At�P   ��m�At�1�   At�P   At�}P   ��m�At���   At�}P   At��P   ��p�At��   At��P   At�_P   ��n�At���   At�_P   At��P   ��m�At���   At��P   At�AP   ��a�At�f�   At�AP   At��P   ��^At���   At��P   At�#P   ��d�At�H�   At�#P   At��P   ��cAt���   At��P   At�P   ��W�At�*�   At�P   At�vP   ��Z'At���   At�vP   At��P   ��X&At��   At��P   At�XP   ��V�At�}�   At�XP   At��P   ��Q�At���   At��P   At�:P   ��Q�At�_�   At�:P   At��P   ��PhAt���   At��P   At�P   ��R"At�A�   At�P   At��P   ��O�At���   At��P   At��P   ��N<At�#�   At��P   At�oP   ��NDAt���   At�oP   At��P   ��Q�At��   At��P   At�QP   ��RSAt�v�   At�QP   At��P   ��N�At���   At��P   At�3P   ��N�At�X�   At�3P   At��P   ��VAt���   At��P   At�P   ��RnAt�:�   At�P   At��P   ��O�At���   At��P   At��P   ��NAt��   At��P   At�hP   ��KPAtō�   At�hP   At��P   ��F�At���   At��P   At�JP   ��D(At�o�   At�JP   At̻P   ��K�At���   At̻P   At�,P   ��D�At�Q�   At�,P   AtѝP   ��E
At���   AtѝP   At�P   ��G�At�3�   At�P   At�P   ��D�At֤�   At�P   At��P   ��J�At��   At��P   At�aP   ��HAtۆ�   At�aP   At��P   ��<NAt���   At��P   At�CP   ��=�At�h�   At�CP   At�P   ��6�At���   At�P   At�%P   ��/At�J�   At�%P   At�P   ��0LAt��   At�P   At�P   ��-�At�,�   At�P   At�xP   ��-bAt��   At�xP   At��P   ��,�At��   At��P   At�ZP   ��1<At��   At�ZP   At��P   ��3�At���   At��P   At�<P   ��1kAt�a�   At�<P   At��P   ��.�At���   At��P   At�P   ��/At�C�   At�P   At��P   ��-�At���   At��P   Au  P   ��*Au %�   Au  P   AuqP   ��.	Au��   AuqP   Au�P   ��1gAu�   Au�P   AuSP   ��,YAux�   AuSP   Au	�P   ��"�Au	��   Au	�P   Au5P   ��'�AuZ�   Au5P   Au�P   ��/�Au��   Au�P   AuP   ��/�Au<�   AuP   Au�P   ��-�Au��   Au�P   Au�P   ��,Au�   Au�P   AujP   ��,MAu��   AujP   Au�P   ��+ZAu �   Au�P   AuLP   ��(�Auq�   AuLP   Au�P   ��+"Au��   Au�P   Au".P   ��.^Au"S�   Au".P   Au$�P   ��)�Au$��   Au$�P   Au'P   ��(�Au'5�   Au'P   Au)�P   ��,1Au)��   Au)�P   Au+�P   ��2�Au,�   Au+�P   Au.cP   ��8�Au.��   Au.cP   Au0�P   ��6�Au0��   Au0�P   Au3EP   ��9Au3j�   Au3EP   Au5�P   ��5IAu5��   Au5�P   Au8'P   ��2)Au8L�   Au8'P   Au:�P   ��9�Au:��   Au:�P   Au=	P   ��6�Au=.�   Au=	P   Au?zP   ��.�Au?��   Au?zP   AuA�P   ��2�AuB�   AuA�P   AuD\P   ��3AuD��   AuD\P   AuF�P   ��4$AuF��   AuF�P   AuI>P   ��4�AuIc�   AuI>P   AuK�P   ��,AuK��   AuK�P   AuN P   ��)2AuNE�   AuN P   AuP�P   ��1lAuP��   AuP�P   AuSP   ��0�AuS'�   AuSP   AuUsP   ��,�AuU��   AuUsP   AuW�P   ��-�AuX	�   AuW�P   AuZUP   ��0[AuZz�   AuZUP   Au\�P   ��+�Au\��   Au\�P   Au_7P   ��0^Au_\�   Au_7P   Aua�P   ��/�Aua��   Aua�P   AudP   ��0Aud>�   AudP   Auf�P   ��'hAuf��   Auf�P   Auh�P   ��,�Aui �   Auh�P   AuklP   ��3Auk��   AuklP   Aum�P   ��-zAun�   Aum�P   AupNP   ��/�Aups�   AupNP   Aur�P   ��0�Aur��   Aur�P   Auu0P   ��3HAuuU�   Auu0P   Auw�P   ��7UAuw��   Auw�P   AuzP   ��:WAuz7�   AuzP   Au|�P   ��6�Au|��   Au|�P   Au~�P   ��3�Au�   Au~�P   Au�eP   ��2�Au���   Au�eP   Au��P   ��/�Au���   Au��P   Au�GP   ��2dAu�l�   Au�GP   Au��P   ��+�Au���   Au��P   Au�)P   ��#�Au�N�   Au�)P   Au��P   ��%rAu���   Au��P   Au�P   ��&�Au�0�   Au�P   Au�|P   ��'�Au���   Au�|P   Au��P   ��!FAu��   Au��P   Au�^P   ��-7Au���   Au�^P   Au��P   ��/Au���   Au��P   Au�@P   ��)�Au�e�   Au�@P   Au��P   ��&DAu���   Au��P   Au�"P   ��$?Au�G�   Au�"P   Au��P   ��(bAu���   Au��P   Au�P   ��(`Au�)�   Au�P   Au�uP   ��.dAu���   Au�uP   Au��P   ��&�Au��   Au��P   Au�WP   ��%&Au�|�   Au�WP   Au��P   ��#TAu���   Au��P   Au�9P   ��&�Au�^�   Au�9P   Au��P   ��#�Au���   Au��P   Au�P   ��)�Au�@�   Au�P   Au��P   ��(�Au���   Au��P   Au��P   ��<Au�"�   Au��P   Au�nP   ��8Au���   Au�nP   Au��P   �� �Au��   Au��P   Au�PP   ���Au�u�   Au�PP   Au��P   ��}Au���   Au��P   Au�2P   ��DAu�W�   Au�2P   AuʣP   ���Au���   AuʣP   Au�P   ���Au�9�   Au�P   AuυP   �� AuϪ�   AuυP   Au��P   �� 0Au��   Au��P   Au�gP   ����AuԌ�   Au�gP   Au��P   ���Au���   Au��P   Au�IP   ����Au�n�   Au�IP   AuۺP   ����Au���   AuۺP   Au�+P   ���dAu�P�   Au�+P   Au��P   ���Au���   Au��P   Au�P   ���Au�2�   Au�P   Au�~P   ���`Au��   Au�~P   Au��P   ����Au��   Au��P   Au�`P   ����Au��   Au�`P   Au��P   ��ܬAu���   Au��P   Au�BP   ���\Au�g�   Au�BP   Au�P   ���Au���   Au�P   Au�$P   ����Au�I�   Au�$P   Au��P   ��ϼAu���   Au��P   Au�P   ����Au�+�   Au�P   Au�wP   ���AAu���   Au�wP   Au��P   ���,Au��   Au��P   Av YP   ���mAv ~�   Av YP   Av�P   ����Av��   Av�P   Av;P   ����Av`�   Av;P   Av�P   ���Av��   Av�P   Av
P   ���Av
B�   Av
P   Av�P   ���9Av��   Av�P   Av�P   ���gAv$�   Av�P   AvpP   ���aAv��   AvpP   Av�P   ����Av�   Av�P   AvRP   ����Avw�   AvRP   Av�P   ���tAv��   Av�P   Av4P   ���AvY�   Av4P   Av�P   ���lAv��   Av�P   Av P   ����Av ;�   Av P   Av"�P   ����Av"��   Av"�P   Av$�P   ���/Av%�   Av$�P   Av'iP   ����Av'��   Av'iP   Av)�P   ����Av)��   Av)�P   Av,KP   ����Av,p�   Av,KP   Av.�P   ����Av.��   Av.�P   Av1-P   ���ZAv1R�   Av1-P   Av3�P   ���9Av3��   Av3�P   Av6P   ����Av64�   Av6P   Av8�P   ����Av8��   Av8�P   Av:�P   ����Av;�   Av:�P   Av=bP   ����Av=��   Av=bP   Av?�P   ����Av?��   Av?�P   AvBDP   ���TAvBi�   AvBDP   AvD�P   ���AvD��   AvD�P   AvG&P   ���AvGK�   AvG&P   AvI�P   ���-AvI��   AvI�P   AvLP   ��{XAvL-�   AvLP   AvNyP   ��u�AvN��   AvNyP   AvP�P   ��uAvQ�   AvP�P   AvS[P   ��o�AvS��   AvS[P   AvU�P   ��x�AvU��   AvU�P   AvX=P   ��}�AvXb�   AvX=P   AvZ�P   ��{<AvZ��   AvZ�P   Av]P   ��xkAv]D�   Av]P   Av_�P   ��xLAv_��   Av_�P   AvbP   ��u�Avb&�   AvbP   AvdrP   ��uAvd��   AvdrP   Avf�P   ��sAvg�   Avf�P   AviTP   ��x�