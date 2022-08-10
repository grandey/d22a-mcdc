CDF  �   
      time       bnds      lon       lat          7   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-ESM2-1 (2017):  aerosol: TACTIC_v2 atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: REPROBUS-C_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) ocnBgchem: Pisces 2.s seaIce: Gelato 6.1    institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    creation_date         2018-04-23T09:02:20Z   description       DECK: control      title         <CNRM-ESM2-1 model output prepared for CMIP6 / CMIP piControl   activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.1    experiment_id         	piControl      
experiment        pre-industrial control     external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-ESM2-1.piControl.none.r1i1p1f2      grid      2native ocean tri-polar grid with 105 k ocean cells     
grid_label        gn     nominal_resolution        100 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      CNRM-ESM2-1    parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent         @�Հ       branch_time_in_child                 physics_index               product       model-output   realization_index               realm         ocean      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         CNRM-ESM2-1    source_type       AOGCM BGC AER CHEM     sub_experiment_id         none   sub_experiment        none   table_id      Omon   variable_id       hfds   variant_info      �. Information provided by this attribute may in some cases be flawed. Users can find more comprehensive and up-to-date documentation via the further_info_url global attribute.    variant_label         r1i1p1f2   EXPID         CNRM-ESM2-1_piControl_r1i1p1f2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          87374385b726e2a5f1e17b33af88ce8c   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.1      history      5Wed Aug 10 15:20:56 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.hfds.gn.v20181115/hfds_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/hfds/CNRM-ESM2-1_r1i1p1f2/hfds_CNRM-ESM2-1_r1i1p1f2_piControl.mergetime.nc
Fri Apr 08 03:09:53 2022: cdo -O -s -selname,hfds -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.hfds.gn.v20181115/hfds_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.hfds.gn.v20181115/hfds_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 03:09:36 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,hfds -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.hfds.gn.v20181115/hfds_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.hfds.gn.v20181115/hfds_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.hfds.gn.v20181115/hfds_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.bic_missto0.yearmean.nc
none      tracking_id       1hdl:21.14100/a6aed716-b08f-404a-8c93-21f5b0f32abf      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   hfds                      standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    online_operation      average    interval_operation        1800 s     interval_write        1 month    description       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any "flux adjustment") .    positive      down   history       none   cell_measures         area: areacello             �                Aq���   Aq��P   Aq�P   �hX�Aq�6�   Aq�P   Aq��P   >�=FAq���   Aq��P   Aq��P   ��S�Aq��   Aq��P   Aq�dP   ?DAq���   Aq�dP   Aq��P   �ҔAq���   Aq��P   Aq�FP   ��(Aq�k�   Aq�FP   Aq��P   >;_Aq���   Aq��P   Aq�(P   ���Aq�M�   Aq�(P   Aq��P   �f�Aq���   Aq��P   Aq�
P   �6*�Aq�/�   Aq�
P   Aq�{P   >�ؤAq���   Aq�{P   Aq��P   ��3sAq��   Aq��P   Aq�]P   >	KAqĂ�   Aq�]P   Aq��P   ���Aq���   Aq��P   Aq�?P   ��5Aq�d�   Aq�?P   Aq˰P   =G.�Aq���   Aq˰P   Aq�!P   >>��Aq�F�   Aq�!P   AqВP   9�.Aqз�   AqВP   Aq�P   <�wxAq�(�   Aq�P   Aq�tP   �,�GAqՙ�   Aq�tP   Aq��P   >��Aq�
�   Aq��P   Aq�VP   �}��Aq�{�   Aq�VP   Aq��P   ���Aq���   Aq��P   Aq�8P   ��Y�Aq�]�   Aq�8P   Aq�P   >	Aq���   Aq�P   Aq�P   ���Aq�?�   Aq�P   Aq�P   >�׾Aq��   Aq�P   Aq��P   ?VgAq�!�   Aq��P   Aq�mP   �Bs�Aq��   Aq�mP   Aq��P   ����Aq��   Aq��P   Aq�OP   >�a3Aq�t�   Aq�OP   Aq��P   =��jAq���   Aq��P   Aq�1P   >`gAq�V�   Aq�1P   Aq��P   >��Aq���   Aq��P   Aq�P   >Gq�Aq�8�   Aq�P   Aq��P   = [�Aq���   Aq��P   Aq��P   >yT�Aq��   Aq��P   ArfP   ���9Ar��   ArfP   Ar�P   >�Ar��   Ar�P   ArHP   �R@�Arm�   ArHP   Ar�P   ��z[Ar��   Ar�P   Ar*P   <h;8ArO�   Ar*P   Ar�P   ���Ar��   Ar�P   ArP   ���Ar1�   ArP   Ar}P   >@!{Ar��   Ar}P   Ar�P   ��3TAr�   Ar�P   Ar_P   >���Ar��   Ar_P   Ar�P   ���Ar��   Ar�P   ArAP   ����Arf�   ArAP   Ar�P   �,2Ar��   Ar�P   Ar!#P   >��Ar!H�   Ar!#P   Ar#�P   �>�Ar#��   Ar#�P   Ar&P   =��PAr&*�   Ar&P   Ar(vP   ��sAr(��   Ar(vP   Ar*�P   >7��Ar+�   Ar*�P   Ar-XP   �2I�Ar-}�   Ar-XP   Ar/�P   =�y,Ar/��   Ar/�P   Ar2:P   =�Z�Ar2_�   Ar2:P   Ar4�P   ?��Ar4��   Ar4�P   Ar7P   =�Ar7A�   Ar7P   Ar9�P   ��aAr9��   Ar9�P   Ar;�P   �c��Ar<#�   Ar;�P   Ar>oP   >$��Ar>��   Ar>oP   Ar@�P   >D�XArA�   Ar@�P   ArCQP   =��ArCv�   ArCQP   ArE�P   �<ܢArE��   ArE�P   ArH3P   >�zArHX�   ArH3P   ArJ�P   >��QArJ��   ArJ�P   ArMP   �JnArM:�   ArMP   ArO�P   >��wArO��   ArO�P   ArQ�P   =iJpArR�   ArQ�P   ArThP   �_��ArT��   ArThP   ArV�P   >��ArV��   ArV�P   ArYJP   >��ArYo�   ArYJP   Ar[�P   =��Ar[��   Ar[�P   Ar^,P   ��"^Ar^Q�   Ar^,P   Ar`�P   �4�Ar`��   Ar`�P   ArcP   �|��Arc3�   ArcP   AreP   =j��Are��   AreP   Arg�P   <spArh�   Arg�P   ArjaP   >��Arj��   ArjaP   Arl�P   =�qwArl��   Arl�P   AroCP   9�kAroh�   AroCP   Arq�P   �ΏArq��   Arq�P   Art%P   >ѾArtJ�   Art%P   Arv�P   ��ۃArv��   Arv�P   AryP   >��Ary,�   AryP   Ar{xP   >#��Ar{��   Ar{xP   Ar}�P   >��Ar~�   Ar}�P   Ar�ZP   =�}�Ar��   Ar�ZP   Ar��P   �Q Ar���   Ar��P   Ar�<P   ?Z4�Ar�a�   Ar�<P   Ar��P   =�z�Ar���   Ar��P   Ar�P   �>^sAr�C�   Ar�P   Ar��P   >��XAr���   Ar��P   Ar� P   �O�nAr�%�   Ar� P   Ar�qP   =�./Ar���   Ar�qP   Ar��P   ���\Ar��   Ar��P   Ar�SP   >�E�Ar�x�   Ar�SP   Ar��P   �c��Ar���   Ar��P   Ar�5P   =�B�Ar�Z�   Ar�5P   Ar��P   ���Ar���   Ar��P   Ar�P   =#!�Ar�<�   Ar�P   Ar��P   ����Ar���   Ar��P   Ar��P   =�)�Ar��   Ar��P   Ar�jP   >S��Ar���   Ar�jP   Ar��P   ��*HAr� �   Ar��P   Ar�LP   ��H�Ar�q�   Ar�LP   Ar��P   >^-�Ar���   Ar��P   Ar�.P   >���Ar�S�   Ar�.P   Ar��P   �+�Ar���   Ar��P   Ar�P   >�Ar�5�   Ar�P   Ar��P   >~qAr���   Ar��P   Ar��P   >A�Ar��   Ar��P   Ar�cP   �o/�Ar���   Ar�cP   Ar��P   >)��Ar���   Ar��P   Ar�EP   >�t�Ar�j�   Ar�EP   ArĶP   �$r�Ar���   ArĶP   Ar�'P   ����Ar�L�   Ar�'P   ArɘP   ><�Arɽ�   ArɘP   Ar�	P   >"7�Ar�.�   Ar�	P   Ar�zP   ��uArΟ�   Ar�zP   Ar��P   �(0Ar��   Ar��P   Ar�\P   �;��ArӁ�   Ar�\P   Ar��P   ?<fdAr���   Ar��P   Ar�>P   <@MAr�c�   Ar�>P   ArگP   ��WAr���   ArگP   Ar� P   >�`�Ar�E�   Ar� P   ArߑP   =d��Ar߶�   ArߑP   Ar�P   >'��Ar�'�   Ar�P   Ar�sP   ;�!Ar��   Ar�sP   Ar��P   �klDAr�	�   Ar��P   Ar�UP   �N�eAr�z�   Ar�UP   Ar��P   >�*Ar���   Ar��P   Ar�7P   >��Ar�\�   Ar�7P   Ar�P   >Z�Ar���   Ar�P   Ar�P   ��>�Ar�>�   Ar�P   Ar��P   >�ѹAr���   Ar��P   Ar��P   >q�Ar� �   Ar��P   Ar�lP   ��#%Ar���   Ar�lP   Ar��P   �+$3Ar��   Ar��P   Ar�NP   =Q�XAr�s�   Ar�NP   As�P   >� �As��   As�P   As0P   >29AsU�   As0P   As�P   >!5KAs��   As�P   As	P   >�JwAs	7�   As	P   As�P   ��:�As��   As�P   As�P   ���\As�   As�P   AseP   >Js�As��   AseP   As�P   ? j�As��   As�P   AsGP   >�aAsl�   AsGP   As�P   <֠�As��   As�P   As)P   �m�AsN�   As)P   As�P   ��As��   As�P   AsP   >�#�As0�   AsP   As!|P   �*KOAs!��   As!|P   As#�P   ����As$�   As#�P   As&^P   �ޓ�As&��   As&^P   As(�P   =U�}As(��   As(�P   As+@P   ��.�As+e�   As+@P   As-�P   >)Y�As-��   As-�P   As0"P   ���MAs0G�   As0"P   As2�P   ���As2��   As2�P   As5P   ?-�|As5)�   As5P   As7uP   �ֆAs7��   As7uP   As9�P   ��k�As:�   As9�P   As<WP   �CSAs<|�   As<WP   As>�P   >m(OAs>��   As>�P   AsA9P   ���AsA^�   AsA9P   AsC�P   =)z�AsC��   AsC�P   AsFP   ���hAsF@�   AsFP   AsH�P   >&AAsH��   AsH�P   AsJ�P   �%yDAsK"�   AsJ�P   AsMnP   >k�YAsM��   AsMnP   AsO�P   ��+�AsP�   AsO�P   AsRPP   >m�zAsRu�   AsRPP   AsT�P   >���AsT��   AsT�P   AsW2P   � g�AsWW�   AsW2P   AsY�P   =�OAsY��   AsY�P   As\P   >8As\9�   As\P   As^�P   ���As^��   As^�P   As`�P   >gYAsa�   As`�P   AscgP   ����Asc��   AscgP   Ase�P   =K�sAse��   Ase�P   AshIP   =��^Ashn�   AshIP   Asj�P   >:u�Asj��   Asj�P   Asm+P   >�m�AsmP�   Asm+P   Aso�P   ��� Aso��   Aso�P   AsrP   >��Asr2�   AsrP   Ast~P   �وFAst��   Ast~P   Asv�P   ���Asw�   Asv�P   Asy`P   >� �Asy��   Asy`P   As{�P   ��{5As{��   As{�P   As~BP   <�yAs~g�   As~BP   As��P   ;���As���   As��P   As�$P   >g;As�I�   As�$P   As��P   ���As���   As��P   As�P   >�qVAs�+�   As�P   As�wP   �ʫcAs���   As�wP   As��P   >8��As��   As��P   As�YP   >`��As�~�   As�YP   As��P   �(�]As���   As��P   As�;P   �� As�`�   As�;P   As��P   ���As���   As��P   As�P   ��_�As�B�   As�P   As��P   ?��As���   As��P   As��P   �ƪ�As�$�   As��P   As�pP   ��>�As���   As�pP   As��P   >V9As��   As��P   As�RP   ��As�w�   As�RP   As��P   ����As���   As��P   As�4P   ><pAs�Y�   As�4P   As��P   ��+As���   As��P   As�P   =`�*As�;�   As�P   As��P   >+1As���   As��P   As��P   =�s�As��   As��P   As�iP   ���As���   As�iP   As��P   >-�As���   As��P   As�KP   >�PAs�p�   As�KP   As��P   ���-As���   As��P   As�-P   >>O5As�R�   As�-P   AsP   �1As���   AsP   As�P   �A�As�4�   As�P   AsǀP   >ݜAsǥ�   AsǀP   As��P   ����As��   As��P   As�bP   =�ڰAṡ�   As�bP   As��P   <��As���   As��P   As�DP   �?�As�i�   As�DP   AsӵP   �D�|As���   AsӵP   As�&P   >�As�K�   As�&P   AsؗP   ��+�Asؼ�   AsؗP   As�P   >��hAs�-�   As�P   As�yP   ���FAsݞ�   As�yP   As��P   ��+WAs��   As��P   As�[P   ��ǬAs��   As�[P   As��P   >�]�As���   As��P   As�=P   ��^*As�b�   As�=P   As�P   ��-As���   As�P   As�P   �υ}As�D�   As�P   As�P   �_&~As��   As�P   As�P   >��/As�&�   As�P   As�rP   �Pz#As��   As�rP   As��P   �αAs��   As��P   As�TP   >尰As�y�   As�TP   As��P   >5��As���   As��P   As�6P   �A,XAs�[�   As�6P   As��P   >g��As���   As��P   AtP   >�3#At=�   AtP   At�P   ����At��   At�P   At�P   ;�̦At�   At�P   At	kP   >�^�At	��   At	kP   At�P   ���At�   At�P   AtMP   ��+Atr�   AtMP   At�P   ��5�At��   At�P   At/P   >�FAtT�   At/P   At�P   >���At��   At�P   AtP   >�u�At6�   AtP   At�P   >v�At��   At�P   At�P   �$�wAt�   At�P   AtdP   ?
�At��   AtdP   At!�P   ��At!��   At!�P   At$FP   =<�At$k�   At$FP   At&�P   ���cAt&��   At&�P   At)(P   >�	9At)M�   At)(P   At+�P   >�ZjAt+��   At+�P   At.
P   �-�At./�   At.
P   At0{P   :nRAt0��   At0{P   At2�P   ?cdAt3�   At2�P   At5]P   �v&yAt5��   At5]P   At7�P   >ћ�At7��   At7�P   At:?P   =�2�At:d�   At:?P   At<�P   >���At<��   At<�P   At?!P   >��At?F�   At?!P   AtA�P   >N�
AtA��   AtA�P   AtDP   ���AtD(�   AtDP   AtFtP   >���AtF��   AtFtP   AtH�P   �+jAtI
�   AtH�P   AtKVP   >�~NAtK{�   AtKVP   AtM�P   >�/&AtM��   AtM�P   AtP8P   �M �AtP]�   AtP8P   AtR�P   >���AtR��   AtR�P   AtUP   �fݩAtU?�   AtUP   AtW�P   �7��AtW��   AtW�P   AtY�P   >���AtZ!�   AtY�P   At\mP   ��[�At\��   At\mP   At^�P   >���At_�   At^�P   AtaOP   >�LAtat�   AtaOP   Atc�P   ��P9Atc��   Atc�P   Atf1P   >y,�AtfV�   Atf1P   Ath�P   =���Ath��   Ath�P   AtkP   >	\;Atk8�   AtkP   Atm�P   =.j{Atm��   Atm�P   Ato�P   >tn,Atp�   Ato�P   AtrfP   >��Atr��   AtrfP   Att�P   <��Att��   Att�P   AtwHP   >�2	Atwm�   AtwHP   Aty�P   =�^nAty��   Aty�P   At|*P   >��_At|O�   At|*P   At~�P   ;�#zAt~��   At~�P   At�P   >�i�At�1�   At�P   At�}P   ��l�At���   At�}P   At��P   =���At��   At��P   At�_P   <�PfAt���   At�_P   At��P   >~V�At���   At��P   At�AP   >�YAt�f�   At�AP   At��P   ��#�At���   At��P   At�#P   �PoAt�H�   At�#P   At��P   >�NOAt���   At��P   At�P   >���At�*�   At�P   At�vP   �BQ�At���   At�vP   At��P   =�QiAt��   At��P   At�XP   =�1%At�}�   At�XP   At��P   ��At���   At��P   At�:P   >�%lAt�_�   At�:P   At��P   �H8At���   At��P   At�P   ;(�WAt�A�   At�P   At��P   ���DAt���   At��P   At��P   =��At�#�   At��P   At�oP   >5WAt���   At�oP   At��P   ��6�At��   At��P   At�QP   <��At�v�   At�QP   At��P   >�AAt���   At��P   At�3P   �w�qAt�X�   At�3P   At��P   <�}At���   At��P   At�P   >�NAt�:�   At�P   At��P   <s��At���   At��P   At��P   >��At��   At��P   At�hP   ��/�Atō�   At�hP   At��P   >�sTAt���   At��P   At�JP   ��At�o�   At�JP   At̻P   =��bAt���   At̻P   At�,P   >K�ZAt�Q�   At�,P   AtѝP   �"��At���   AtѝP   At�P   >f�.At�3�   At�P   At�P   =�KaAt֤�   At�P   At��P   ��{�At��   At��P   At�aP   >�w?Atۆ�   At�aP   At��P   >���At���   At��P   At�CP   �%k�At�h�   At�CP   At�P   >О�At���   At�P   At�%P   >�2|At�J�   At�%P   At�P   ��ƥAt��   At�P   At�P   >�At�,�   At�P   At�xP   �W5At��   At�xP   At��P   =\l�At��   At��P   At�ZP   ���At��   At�ZP   At��P   �S��At���   At��P   At�<P   >��)At�a�   At�<P   At��P   ��-�At���   At��P   At�P   =���At�C�   At�P   At��P   >���At���   At��P   Au  P   ���Au %�   Au  P   AuqP   �6��Au��   AuqP   Au�P   �#��Au�   Au�P   AuSP   ?gAux�   AuSP   Au	�P   �!��Au	��   Au	�P   Au5P   ��&AuZ�   Au5P   Au�P   �U|UAu��   Au�P   AuP   >��Au<�   AuP   Au�P   <�Au��   Au�P   Au�P   =�8Au�   Au�P   AujP   =_��Au��   AujP   Au�P   ���Au �   Au�P   AuLP   >\aFAuq�   AuLP   Au�P   ��§Au��   Au�P   Au".P   <ΝCAu"S�   Au".P   Au$�P   =�wAu$��   Au$�P   Au'P   ��%�Au'5�   Au'P   Au)�P   �q�BAu)��   Au)�P   Au+�P   ���Au,�   Au+�P   Au.cP   �+�xAu.��   Au.cP   Au0�P   >-a�Au0��   Au0�P   Au3EP   >F�(Au3j�   Au3EP   Au5�P   =���Au5��   Au5�P   Au8'P   �O.Au8L�   Au8'P   Au:�P   ���GAu:��   Au:�P   Au=	P   >4>-Au=.�   Au=	P   Au?zP   =8�Au?��   Au?zP   AuA�P   ��]�AuB�   AuA�P   AuD\P   �0� AuD��   AuD\P   AuF�P   �Ё1AuF��   AuF�P   AuI>P   >.�nAuIc�   AuI>P   AuK�P   >���AuK��   AuK�P   AuN P   ��55AuNE�   AuN P   AuP�P   �`�AuP��   AuP�P   AuSP   �QJ�AuS'�   AuSP   AuUsP   =��AuU��   AuUsP   AuW�P   =�v�AuX	�   AuW�P   AuZUP   ���AuZz�   AuZUP   Au\�P   =�bAu\��   Au\�P   Au_7P   ��ޠAu_\�   Au_7P   Aua�P   =�Aua��   Aua�P   AudP   =x�Aud>�   AudP   Auf�P   =�W�Auf��   Auf�P   Auh�P   ��hAui �   Auh�P   AuklP   >,�6Auk��   AuklP   Aum�P   >���Aun�   Aum�P   AupNP   ��AAups�   AupNP   Aur�P   >�qAur��   Aur�P   Auu0P   ��AuuU�   Auu0P   Auw�P   =���Auw��   Auw�P   AuzP   ���Auz7�   AuzP   Au|�P   >���Au|��   Au|�P   Au~�P   <3rVAu�   Au~�P   Au�eP   >�q'Au���   Au�eP   Au��P   ��?�Au���   Au��P   Au�GP   ��%Au�l�   Au�GP   Au��P   >�)yAu���   Au��P   Au�)P   =v8�Au�N�   Au�)P   Au��P   �|V6Au���   Au��P   Au�P   �XpAu�0�   Au�P   Au�|P   >�|�Au���   Au�|P   Au��P   �шAu��   Au��P   Au�^P   �ZAu���   Au�^P   Au��P   =���Au���   Au��P   Au�@P   ?IsAu�e�   Au�@P   Au��P   �]�Au���   Au��P   Au�"P   >�AAu�G�   Au�"P   Au��P   �h=�Au���   Au��P   Au�P   =��^Au�)�   Au�P   Au�uP   �^�Au���   Au�uP   Au��P   ?1��Au��   Au��P   Au�WP   ��~�Au�|�   Au�WP   Au��P   =�	zAu���   Au��P   Au�9P   ��&Au�^�   Au�9P   Au��P   =�	�Au���   Au��P   Au�P   ���2Au�@�   Au�P   Au��P   >���Au���   Au��P   Au��P   ? �Au�"�   Au��P   Au�nP   �ﺛAu���   Au�nP   Au��P   =�6OAu��   Au��P   Au�PP   >��Au�u�   Au�PP   Au��P   �RAu���   Au��P   Au�2P   >��MAu�W�   Au�2P   AuʣP   >(Au���   AuʣP   Au�P   ?MAu�9�   Au�P   AuυP   >11AuϪ�   AuυP   Au��P   >��?Au��   Au��P   Au�gP   ��?AuԌ�   Au�gP   Au��P   =鼍Au���   Au��P   Au�IP   >�RVAu�n�   Au�IP   AuۺP   =��jAu���   AuۺP   Au�+P   <���Au�P�   Au�+P   Au��P   ?*4Au���   Au��P   Au�P   ����Au�2�   Au�P   Au�~P   >��fAu��   Au�~P   Au��P   =P��Au��   Au��P   Au�`P   >�5uAu��   Au�`P   Au��P   ��wPAu���   Au��P   Au�BP   >�R�Au�g�   Au�BP   Au�P   >5yGAu���   Au�P   Au�$P   �QYhAu�I�   Au�$P   Au��P   >��Au���   Au��P   Au�P   <�?�Au�+�   Au�P   Au�wP   >���Au���   Au�wP   Au��P   ���dAu��   Au��P   Av YP   >W&�Av ~�   Av YP   Av�P   >DtRAv��   Av�P   Av;P   ?	�Av`�   Av;P   Av�P   ���%Av��   Av�P   Av
P   >È�Av
B�   Av
P   Av�P   �#'�Av��   Av�P   Av�P   >�ەAv$�   Av�P   AvpP   �`Av��   AvpP   Av�P   =��Av�   Av�P   AvRP   >�@Avw�   AvRP   Av�P   ?	�wAv��   Av�P   Av4P   �*��AvY�   Av4P   Av�P   >��0Av��   Av�P   Av P   ���Av ;�   Av P   Av"�P   >�k�Av"��   Av"�P   Av$�P   >|G�Av%�   Av$�P   Av'iP   >s�YAv'��   Av'iP   Av)�P   >
kAv)��   Av)�P   Av,KP   >�x�Av,p�   Av,KP   Av.�P   ��Y�Av.��   Av.�P   Av1-P   >�r�Av1R�   Av1-P   Av3�P   >��Av3��   Av3�P   Av6P   =,�{Av64�   Av6P   Av8�P   >�IAv8��   Av8�P   Av:�P   ��g�Av;�   Av:�P   Av=bP   =���Av=��   Av=bP   Av?�P   ��e�Av?��   Av?�P   AvBDP   >���AvBi�   AvBDP   AvD�P   ;�%�AvD��   AvD�P   AvG&P   >���AvGK�   AvG&P   AvI�P   >�TAvI��   AvI�P   AvLP   >���AvL-�   AvLP   AvNyP   �$��AvN��   AvNyP   AvP�P   >��AvQ�   AvP�P   AvS[P   ���AvS��   AvS[P   AvU�P   ���OAvU��   AvU�P   AvX=P   �-CAvXb�   AvX=P   AvZ�P   >�VAvZ��   AvZ�P   Av]P   ���Av]D�   Av]P   Av_�P   ��� Av_��   Av_�P   AvbP   �B
Avb&�   AvbP   AvdrP   =B��Avd��   AvdrP   Avf�P   > ��Avg�   Avf�P   AviTP   ����