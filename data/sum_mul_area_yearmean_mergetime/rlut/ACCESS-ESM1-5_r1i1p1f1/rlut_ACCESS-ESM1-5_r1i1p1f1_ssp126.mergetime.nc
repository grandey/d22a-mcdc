CDF  �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       oACCESS-ESM1.5 (2019): 
aerosol: CLASSIC (v1.0)
atmos: HadGAM2 (r1.1, N96; 192 x 145 longitude/latitude; 38 levels; top level 39255 m)
atmosChem: none
land: CABLE2.4
landIce: none
ocean: ACCESS-OM2 (MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: WOMBAT (same grid as ocean)
seaIce: CICE4.1 (same grid as ocean)    institution       aCommonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         @�f�       creation_date         2019-11-15T06:37:46Z   data_specs_version        01.00.30   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Qhttps://furtherinfo.es-doc.org/CMIP6.CSIRO.ACCESS-ESM1-5.historical.none.r1i1p1f1      grid      ,native atmosphere N96 grid (145x192 latxlon)   
grid_label        gn     history      �Wed Nov 09 19:00:38 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/ACCESS-ESM1-5_r1i1p1f1/rlut_ACCESS-ESM1-5_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/ACCESS-ESM1-5_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rlut.gn.v20210318/rlut_Amon_ACCESS-ESM1-5_ssp126_r1i1p1f1_gn_201501-210012.yearmean.mul.areacella_ssp126_v20210318.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/ACCESS-ESM1-5_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rlut.gn.v20210318/rlut_Amon_ACCESS-ESM1-5_ssp126_r1i1p1f1_gn_210101-230012.yearmean.mul.areacella_ssp126_v20210318.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/ACCESS-ESM1-5_r1i1p1f1/rlut_ACCESS-ESM1-5_r1i1p1f1_ssp126.mergetime.nc
Wed Nov 09 19:00:37 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.rlut.gn.v20191115/rlut_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20191115.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/ACCESS-ESM1-5_r1i1p1f1/rlut_ACCESS-ESM1-5_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 02:30:02 2022: cdo -O -s -fldsum -setattribute,rlut@units=W m-2 m2 -mul -yearmean -selname,rlut /Users/benjamin/Data/p22b/CMIP6/rlut/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.rlut.gn.v20191115/rlut_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.fx.areacella.gn.v20191115/areacella_fx_ACCESS-ESM1-5_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/ACCESS-ESM1-5_r1i1p1f1/CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.rlut.gn.v20191115/rlut_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20191115.fldsum.nc
2019-11-15T06:37:46Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.    initialization_index            institution_id        CSIRO      mip_era       CMIP6      nominal_resolution        250 km     notes         FExp: ESM-historical; Local ID: HI-05; Variable: rlut (['fld_s03i332'])     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      ACCESS-ESM1-5      parent_time_units         days since 0101-1-1    parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         ACCESS-ESM1-5      source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         'ACCESS-ESM1-5 output prepared for CMIP6    variable_id       rlut   variant_label         r1i1p1f1   version       	v20191115      cmor_version      3.4.0      tracking_id       1hdl:21.14100/63549b68-d952-4e39-a7c1-3313269e93aa      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               D   	time_bnds                                 L   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               4   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               <   rlut                   	   standard_name         toa_outgoing_longwave_flux     	long_name         TOA Outgoing Longwave Radiation    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       Iat the top of the atmosphere (to be compared with satellite measurements)      cell_measures         area: areacella    history       u2019-11-15T06:37:44Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               \                Aq���   Aq��P   Aq�P   [�y�Aq�6�   Aq�P   Aq��P   [�>�Aq���   Aq��P   Aq��P   [�Aq��   Aq��P   Aq�dP   [��Aq���   Aq�dP   Aq��P   [�9�Aq���   Aq��P   Aq�FP   [�.fAq�k�   Aq�FP   Aq��P   [�9qAq���   Aq��P   Aq�(P   [�'6Aq�M�   Aq�(P   Aq��P   [��Aq���   Aq��P   Aq�
P   [�KAq�/�   Aq�
P   Aq�{P   [���Aq���   Aq�{P   Aq��P   [���Aq��   Aq��P   Aq�]P   [ݾ�AqĂ�   Aq�]P   Aq��P   [�
�Aq���   Aq��P   Aq�?P   [�S�Aq�d�   Aq�?P   Aq˰P   [��WAq���   Aq˰P   Aq�!P   [�=xAq�F�   Aq�!P   AqВP   [��Aqз�   AqВP   Aq�P   [�Z�Aq�(�   Aq�P   Aq�tP   [�R�Aqՙ�   Aq�tP   Aq��P   [�XAq�
�   Aq��P   Aq�VP   [�F�Aq�{�   Aq�VP   Aq��P   [�.�Aq���   Aq��P   Aq�8P   [��Aq�]�   Aq�8P   Aq�P   [ީ�Aq���   Aq�P   Aq�P   [ޛ�Aq�?�   Aq�P   Aq�P   [�<7Aq��   Aq�P   Aq��P   [�V�Aq�!�   Aq��P   Aq�mP   [�'�Aq��   Aq�mP   Aq��P   [�XAq��   Aq��P   Aq�OP   [�XAq�t�   Aq�OP   Aq��P   [��Aq���   Aq��P   Aq�1P   [�Aq�V�   Aq�1P   Aq��P   [��1Aq���   Aq��P   Aq�P   [ܞmAq�8�   Aq�P   Aq��P   [�>�Aq���   Aq��P   Aq��P   [ݿ.Aq��   Aq��P   ArfP   [��,Ar��   ArfP   Ar�P   [��lAr��   Ar�P   ArHP   [�ZArm�   ArHP   Ar�P   [��Ar��   Ar�P   Ar*P   [ݺ�ArO�   Ar*P   Ar�P   [��5Ar��   Ar�P   ArP   [�BAr1�   ArP   Ar}P   [��UAr��   Ar}P   Ar�P   [�2�Ar�   Ar�P   Ar_P   [��VAr��   Ar_P   Ar�P   [���Ar��   Ar�P   ArAP   [�6�Arf�   ArAP   Ar�P   [�*pAr��   Ar�P   Ar!#P   [�O�Ar!H�   Ar!#P   Ar#�P   [�\Ar#��   Ar#�P   Ar&P   [ݻ�Ar&*�   Ar&P   Ar(vP   [�1tAr(��   Ar(vP   Ar*�P   [݊�Ar+�   Ar*�P   Ar-XP   [ݓ5Ar-}�   Ar-XP   Ar/�P   [ݿ�Ar/��   Ar/�P   Ar2:P   [ݿ�Ar2_�   Ar2:P   Ar4�P   [��Ar4��   Ar4�P   Ar7P   [��iAr7A�   Ar7P   Ar9�P   [��Ar9��   Ar9�P   Ar;�P   [�D/Ar<#�   Ar;�P   Ar>oP   [ݫ�Ar>��   Ar>oP   Ar@�P   [��ArA�   Ar@�P   ArCQP   [��ArCv�   ArCQP   ArE�P   [�2�ArE��   ArE�P   ArH3P   [�L[ArHX�   ArH3P   ArJ�P   [��ArJ��   ArJ�P   ArMP   [�0�ArM:�   ArMP   ArO�P   [ݸ:ArO��   ArO�P   ArQ�P   [�|MArR�   ArQ�P   ArThP   [��ArT��   ArThP   ArV�P   [�x�ArV��   ArV�P   ArYJP   [ݫ�ArYo�   ArYJP   Ar[�P   [��
Ar[��   Ar[�P   Ar^,P   [��Ar^Q�   Ar^,P   Ar`�P   [�5dAr`��   Ar`�P   ArcP   [��%Arc3�   ArcP   AreP   [���Are��   AreP   Arg�P   [�~�Arh�   Arg�P   ArjaP   [ݾ�Arj��   ArjaP   Arl�P   [ݻ�Arl��   Arl�P   AroCP   [��oAroh�   AroCP   Arq�P   [�sArq��   Arq�P   Art%P   [ݜ�ArtJ�   Art%P   Arv�P   [�ՏArv��   Arv�P   AryP   [ݦLAry,�   AryP   Ar{xP   [��-Ar{��   Ar{xP   Ar}�P   [���Ar~�   Ar}�P   Ar�ZP   [ݯ�Ar��   Ar�ZP   Ar��P   [ݧ@Ar���   Ar��P   Ar�<P   [ݣ�Ar�a�   Ar�<P   Ar��P   [ݛTAr���   Ar��P   Ar�P   [�Ar�C�   Ar�P   Ar��P   [��Ar���   Ar��P   Ar� P   [ݿAr�%�   Ar� P   Ar�qP   [ݠ�Ar���   Ar�qP   Ar��P   [ݨ�Ar��   Ar��P   Ar�SP   [��|Ar�x�   Ar�SP   Ar��P   [���Ar���   Ar��P   Ar�5P   [�O+Ar�Z�   Ar�5P   Ar��P   [ݏ�Ar���   Ar��P   Ar�P   [�'�Ar�<�   Ar�P   Ar��P   [�a�Ar���   Ar��P   Ar��P   [ݏ�Ar��   Ar��P   Ar�jP   [���Ar���   Ar�jP   Ar��P   [�g8Ar� �   Ar��P   Ar�LP   [�l�Ar�q�   Ar�LP   Ar��P   [݄Ar���   Ar��P   Ar�.P   [�jSAr�S�   Ar�.P   Ar��P   [�GAr���   Ar��P   Ar�P   [�+�Ar�5�   Ar�P   Ar��P   [ݑ5Ar���   Ar��P   Ar��P   [�TnAr��   Ar��P   Ar�cP   [ܨ�Ar���   Ar�cP   Ar��P   [�ڠAr���   Ar��P   Ar�EP   [�,:Ar�j�   Ar�EP   ArĶP   [ܘ�Ar���   ArĶP   Ar�'P   [�׷Ar�L�   Ar�'P   ArɘP   [�8Arɽ�   ArɘP   Ar�	P   [�<�Ar�.�   Ar�	P   Ar�zP   [���ArΟ�   Ar�zP   Ar��P   [��Ar��   Ar��P   Ar�\P   [�*ArӁ�   Ar�\P   Ar��P   [��Ar���   Ar��P   Ar�>P   [ܰAr�c�   Ar�>P   ArگP   [���Ar���   ArگP   Ar� P   [�SAr�E�   Ar� P   ArߑP   [���Ar߶�   ArߑP   Ar�P   [���Ar�'�   Ar�P   Ar�sP   [�K�Ar��   Ar�sP   Ar��P   [��!Ar�	�   Ar��P   Ar�UP   [�q�Ar�z�   Ar�UP   Ar��P   [�c�Ar���   Ar��P   Ar�7P   [ܿ�Ar�\�   Ar�7P   Ar�P   [�v�Ar���   Ar�P   Ar�P   [�f�Ar�>�   Ar�P   Ar��P   [܉�Ar���   Ar��P   Ar��P   [ܾ�Ar� �   Ar��P   Ar�lP   [��Ar���   Ar�lP   Ar��P   [��Ar��   Ar��P   Ar�NP   [�zbAr�s�   Ar�NP   As�P   [�mAs��   As�P   As0P   [� �AsU�   As0P   As�P   [�G�As��   As�P   As	P   [܍uAs	7�   As	P   As�P   [� �As��   As�P   As�P   [���As�   As�P   AseP   [�\As��   AseP   As�P   [�qAs��   As�P   AsGP   [�:^Asl�   AsGP   As�P   [�~pAs��   As�P   As)P   [��'AsN�   As)P   As�P   [��OAs��   As�P   AsP   [ݨ�As0�   AsP   As!|P   [ݲ�As!��   As!|P   As#�P   [�sPAs$�   As#�P   As&^P   [� zAs&��   As&^P   As(�P   [�{�As(��   As(�P   As+@P   [ݭ�As+e�   As+@P   As-�P   [��As-��   As-�P   As0"P   [݌JAs0G�   As0"P   As2�P   [�zAs2��   As2�P   As5P   [ݤ(As5)�   As5P   As7uP   [ݝAs7��   As7uP   As9�P   [��As:�   As9�P   As<WP   [�o�As<|�   As<WP   As>�P   [�y�As>��   As>�P   AsA9P   [�b�AsA^�   AsA9P   AsC�P   [�uAsC��   AsC�P   AsFP   [��gAsF@�   AsFP   AsH�P   [�s�AsH��   AsH�P   AsJ�P   [�"�AsK"�   AsJ�P   AsMnP   [��AsM��   AsMnP   AsO�P   [�+�AsP�   AsO�P   AsRPP   [�$�AsRu�   AsRPP   AsT�P   [��%AsT��   AsT�P   AsW2P   [�a�AsWW�   AsW2P   AsY�P   [�IAsY��   AsY�P   As\P   [�]�As\9�   As\P   As^�P   [��As^��   As^�P   As`�P   [�T�Asa�   As`�P   AscgP   [�,�Asc��   AscgP   Ase�P   [�!�Ase��   Ase�P   AshIP   [�61Ashn�   AshIP   Asj�P   [ޖ'Asj��   Asj�P   Asm+P   [�#�AsmP�   Asm+P   Aso�P   [ޒtAso��   Aso�P   AsrP   [�WEAsr2�   AsrP   Ast~P   [��Ast��   Ast~P   Asv�P   [�yAsw�   Asv�P   Asy`P   [�c�Asy��   Asy`P   As{�P   [ޭAs{��   As{�P   As~BP   [���As~g�   As~BP   As��P   [ޞ>As���   As��P   As�$P   [ހ_As�I�   As�$P   As��P   [��[As���   As��P   As�P   [ދ�As�+�   As�P   As�wP   [ޜlAs���   As�wP   As��P   [���As��   As��P   As�YP   [�_�As�~�   As�YP   As��P   [޵�As���   As��P   As�;P   [���As�`�   As�;P   As��P   [��As���   As��P   As�P   [��[As�B�   As�P   As��P   [ފMAs���   As��P   As��P   [ޤsAs�$�   As��P   As�pP   [���As���   As�pP   As��P   [�2�As��   As��P   As�RP   [�0SAs�w�   As�RP   As��P   [�� As���   As��P   As�4P   [��As�Y�   As�4P   As��P   [�^As���   As��P   As�P   [��As�;�   As�P   As��P   [��As���   As��P   As��P   [޳oAs��   As��P   As�iP   [���As���   As�iP   As��P   [�gAs���   As��P   As�KP   [�"As�p�   As�KP   As��P   [߀�As���   As��P   As�-P   [�As�R�   As�-P   AsP   [ߎ]As���   AsP   As�P   [�.�As�4�   As�P   AsǀP   [�_�Asǥ�   AsǀP   As��P   [�m�As��   As��P   As�bP   [�tAṡ�   As�bP   As��P   [�d�As���   As��P   As�DP   [�TiAs�i�   As�DP   AsӵP   [�t As���   AsӵP   As�&P   [�ukAs�K�   As�&P   AsؗP   [߇<Asؼ�   AsؗP   As�P   [�S�As�-�   As�P   As�yP   [ߔ�Asݞ�   As�yP   As��P   [�zAs��   As��P   As�[P   [߶As��   As�[P   As��P   [ߝWAs���   As��P   As�=P   [��FAs�b�   As�=P   As�P   [߀uAs���   As�P   As�P   [�S�As�D�   As�P   As�P   [ߤAs��   As�P   As�P   [ߜIAs�&�   As�P   As�rP   [߬�As��   As�rP   As��P   [��)As��   As��P   As�TP   [�gAs�y�   As�TP   As��P   [�OAs���   As��P   As�6P   [߰�As�[�   As�6P   As��P   [���As���   As��P   AtP   [ߩAt=�   AtP   At�P   [�n�At��   At�P   At�P   [���At�   At�P   At	kP   [ߴAt	��   At	kP   At�P   [���At�   At�P   AtMP   [ߠ�Atr�   AtMP   At�P   [߬�At��   At�P   At/P   [ߝ�AtT�   At/P   At�P   [߱�At��   At�P   AtP   [߇At6�   AtP   At�P   [�|)At��   At�P   At�P   [ߏ�At�   At�P   AtdP   [�At��   AtdP   At!�P   [�|3At!��   At!�P   At$FP   [��At$k�   At$FP   At&�P   [�ޓAt&��   At&�P   At)(P   [߭�At)M�   At)(P   At+�P   [�ֲAt+��   At+�P   At.
P   [�_TAt./�   At.
P   At0{P   [�x3At0��   At0{P   At2�P   [߿At3�   At2�P   At5]P   [ߙ�At5��   At5]P   At7�P   [�T:At7��   At7�P   At:?P   [��At:d�   At:?P   At<�P   [ߚ�At<��   At<�P   At?!P   [��At?F�   At?!P   AtA�P   [��AtA��   AtA�P   AtDP   [�OeAtD(�   AtDP   AtFtP   [��AtF��   AtFtP   AtH�P   [�%sAtI
�   AtH�P   AtKVP   [��AtK{�   AtKVP   AtM�P   [�ƖAtM��   AtM�P   AtP8P   [��AtP]�   AtP8P   AtR�P   [��`AtR��   AtR�P   AtUP   [��GAtU?�   AtUP   AtW�P   [�~AtW��   AtW�P   AtY�P   [�9SAtZ!�   AtY�P   At\mP   [��+At\��   At\mP   At^�P   [��fAt_�   At^�P   AtaOP   [�żAtat�   AtaOP   Atc�P   [��!Atc��   Atc�P   Atf1P   [���AtfV�   Atf1P   Ath�P   [�u7Ath��   Ath�P   AtkP   [ߗAtk8�   AtkP   Atm�P   [�	9Atm��   Atm�P   Ato�P   [��cAtp�   Ato�P   AtrfP   [ߩAtr��   AtrfP   Att�P   [��Att��   Att�P   AtwHP   [��Atwm�   AtwHP   Aty�P   [߮�Aty��   Aty�P   At|*P   [��At|O�   At|*P   At~�P   [ߦ�At~��   At~�P   At�P   [��gAt�1�   At�P   At�}P   [ߍ�At���   At�}P   At��P   [�q[At��   At��P   At�_P   [ߢ/At���   At�_P   At��P   [�~RAt���   At��P   At�AP   [߸sAt�f�   At�AP   At��P   [��At���   At��P   At�#P   [��At�H�   At�#P   At��P   [ߙ�At���   At��P   At�P   [�ԧAt�*�   At�P   At�vP   [ߋ6At���   At�vP   At��P   [ߒ&At��   At��P   At�XP   [߸dAt�}�   At�XP   At��P   [�cmAt���   At��P   At�:P   [���At�_�   At�:P   At��P   [ߢAt���   At��P   At�P   [���At�A�   At�P   At��P   [߀�At���   At��P   At��P   [�N At�#�   At��P   At�oP   [�RAt���   At�oP   At��P   [�!gAt��   At��P   At�QP   [���At�v�   At�QP   At��P   [߆�At���   At��P   At�3P   [߳�At�X�   At�3P   At��P   [���At���   At��P   At�P   [�,YAt�:�   At�P   At��P   [�TAt���   At��P   At��P   [߲�At��   At��P   At�hP   [��5Atō�   At�hP   At��P   [ߵUAt���   At��P   At�JP   [ߋwAt�o�   At�JP   At̻P   [�ԝAt���   At̻P   At�,P   [ߙ�At�Q�   At�,P   AtѝP   [��?At���   AtѝP   At�P   [ߣuAt�3�   At�P   At�P   [� &At֤�   At�P   At��P   [��UAt��   At��P   At�aP   [���Atۆ�   At�aP   At��P   [�'SAt���   At��P   At�CP   [���At�h�   At�CP   At�P   [ߘ�At���   At�P   At�%P   [��At�J�   At�%P   At�P   [���At��   At�P   At�P   [�o�At�,�   At�P   At�xP   [�1At��   At�xP   At��P   [� �At��   At��P   At�ZP   [��At��   At�ZP   At��P   [��BAt���   At��P   At�<P   [��>At�a�   At�<P   At��P   [�'�At���   At��P   At�P   [���At�C�   At�P   At��P   [��-At���   At��P   Au  P   [�fAu %�   Au  P   AuqP   [߭�Au��   AuqP   Au�P   [��TAu�   Au�P   AuSP   [��PAux�   AuSP   Au	�P   [ߘ�Au	��   Au	�P   Au5P   [��9AuZ�   Au5P   Au�P   [ߛ�Au��   Au�P   AuP   [� Au<�   AuP   Au�P   [�U�Au��   Au�P   Au�P   [߳�Au�   Au�P   AujP   [��]Au��   AujP   Au�P   [�p�Au �   Au�P   AuLP   [�[�Auq�   AuLP   Au�P   [߲Au��   Au�P   Au".P   [���Au"S�   Au".P   Au$�P   [�UAu$��   Au$�P   Au'P   [��Au'5�   Au'P   Au)�P   [���Au)��   Au)�P   Au+�P   [��Au,�   Au+�P   Au.cP   [���Au.��   Au.cP   Au0�P   [߾:Au0��   Au0�P   Au3EP   [��fAu3j�   Au3EP   Au5�P   [�	�Au5��   Au5�P   Au8'P   [�g�Au8L�   Au8'P   Au:�P   [�J�Au:��   Au:�P   Au=	P   [��Au=.�   Au=	P   Au?zP   [��Au?��   Au?zP   AuA�P   [��zAuB�   AuA�P   AuD\P   [� }AuD��   AuD\P   AuF�P   [�2AuF��   AuF�P   AuI>P   [� bAuIc�   AuI>P   AuK�P   [ߎyAuK��   AuK�P   AuN P   [ߍdAuNE�   AuN P   AuP�P   [��,AuP��   AuP�P   AuSP   [߂�AuS'�   AuSP   AuUsP   [ߴ?AuU��   AuUsP   AuW�P   [��AuX	�   AuW�P   AuZUP   [��wAuZz�   AuZUP   Au\�P   [�ӟAu\��   Au\�P   Au_7P   [��Au_\�   Au_7P   Aua�P   [�=Aua��   Aua�P   AudP   [��{Aud>�   AudP   Auf�P   [߭�Auf��   Auf�P   Auh�P   [߭�Aui �   Auh�P   AuklP   [���Auk��   AuklP   Aum�P   [��Aun�   Aum�P   AupNP   [���Aups�   AupNP   Aur�P   [���Aur��   Aur�P   Auu0P   [߲�AuuU�   Auu0P   Auw�P   [�aAuw��   Auw�P   AuzP   [��Auz7�   AuzP   Au|�P   [���Au|��   Au|�P   Au~�P   [��Au�   Au~�P   Au�eP   [�HsAu���   Au�eP   Au��P   [�/Au���   Au��P   Au�GP   [�>Au�l�   Au�GP   Au��P   [��?Au���   Au��P   Au�)P   [��Au�N�   Au�)P   Au��P   [�3Au���   Au��P   Au�P   [��Au�0�   Au�P   Au�|P   [��Au���   Au�|P   Au��P   [�%Au��   Au��P   Au�^P   [ߺ�Au���   Au�^P   Au��P   [߆�Au���   Au��P   Au�@P   [��Au�e�   Au�@P   Au��P   [���Au���   Au��P   Au�"P   [���Au�G�   Au�"P   Au��P   [ߴ�Au���   Au��P   Au�P   [ߗ:Au�)�   Au�P   Au�uP   [���Au���   Au�uP   Au��P   [�@�Au��   Au��P   Au�WP   [ߢeAu�|�   Au�WP   Au��P   [��RAu���   Au��P   Au�9P   [��Au�^�   Au�9P   Au��P   [��$Au���   Au��P   Au�P   [��Au�@�   Au�P   Au��P   [�m�Au���   Au��P   Au��P   [�)Au�"�   Au��P   Au�nP   [�Q�Au���   Au�nP   Au��P   [��Au��   Au��P   Au�PP   [��!Au�u�   Au�PP   Au��P   [��hAu���   Au��P   Au�2P   [�y�Au�W�   Au�2P   AuʣP   [��<Au���   AuʣP   Au�P   [�+�Au�9�   Au�P   AuυP   [ߧCAuϪ�   AuυP   Au��P   [���Au��   Au��P   Au�gP   [��%AuԌ�   Au�gP   Au��P   [�?�Au���   Au��P   Au�IP   [��iAu�n�   Au�IP   AuۺP   [�WAu���   AuۺP   Au�+P   [��GAu�P�   Au�+P   Au��P   [߾�Au���   Au��P   Au�P   [��Au�2�   Au�P   Au�~P   [ߜ�Au��   Au�~P   Au��P   [߉�Au��   Au��P   Au�`P   [߼{Au��   Au�`P   Au��P   [ߓqAu���   Au��P   Au�BP   [�ԸAu�g�   Au�BP   Au�P   [ߠh